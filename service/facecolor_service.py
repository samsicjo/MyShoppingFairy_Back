# face_color_extraction_refactored.py
# 얼굴 부위별 색상 추출 - Segformer 기반, HEX 코드 추출 기능

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fastapi import UploadFile, HTTPException
import io
import os
from typing import List, Dict, Any, Optional
import colorsys

class FaceColorExtractor:
    def __init__(self):
        """얼굴 파싱 모델 초기화"""
        self.processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        
        self.label_map = {
            0: "background", 1: "skin", 2: "nose", 3: "eye_g", 4: "left_eye",
            5: "right_eye", 6: "left_brow", 7: "right_brow", 8: "left_ear", 9: "right_ear",
            10: "mouth", 11: "upper_lip", 12: "lower_lip", 13: "hair", 14: "hat",
            15: "earr_l", 16: "earr_r", 17: "neck_l", 18: "neck"
        }
        
        self.target_parts = {
            "skin": [1], "nose": [2], "hair": [13], "eyes": [4, 5], "lips": [10, 11, 12]
        }

    def validate_and_count_faces(self, segmentation_mask, original_image_shape):
        """세그멘테이션 마스크를 사용하여 유효한 단일 얼굴이 있는지 검증합니다."""
        unique_labels = np.unique(segmentation_mask)
        has_skin = 1 in unique_labels
        has_eye = 4 in unique_labels or 5 in unique_labels

        if not (has_skin and has_eye):
            raise HTTPException(status_code=400, detail="얼굴의 핵심 부위(피부, 눈)가 인식되지 않았습니다. 더 선명한 사진을 사용해주세요.")

        skin_mask = (segmentation_mask == 1).astype(np.uint8)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_face_area = (original_image_shape[0] * original_image_shape[1]) * 0.01
        face_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_face_area]

        if len(face_contours) == 0:
            raise HTTPException(status_code=400, detail="얼굴을 찾을 수 없습니다. 조명이 밝고 얼굴이 잘 보이는 사진을 사용해주세요.")
        if len(face_contours) > 1:
            raise HTTPException(status_code=400, detail=f"{len(face_contours)}명의 얼굴이 감지되었습니다. 한 명의 얼굴만 있는 사진을 사용해주세요.")

        face_area = cv2.contourArea(face_contours[0])
        image_area = original_image_shape[0] * original_image_shape[1]
        face_ratio = face_area / image_area
        if face_ratio < 0.03:
            raise HTTPException(status_code=400, detail="얼굴이 너무 작습니다. 더 가까이 찍은 사진을 사용해주세요.")
        print(f"얼굴 검증 완료: 1개의 얼굴 감지, 얼굴 비율: {face_ratio:.2%}")

    def parse_face_from_memory(self, image):
        """메모리의 이미지에서 얼굴을 파싱하고 검증합니다."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_segmentation = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].numpy()
        self.validate_and_count_faces(predicted_segmentation, np.array(image).shape)
        return np.array(image), predicted_segmentation

    def _extract_sorted_dominant_colors(self, pixels: np.ndarray, n_colors: int = 1) -> Optional[List[np.ndarray]]:
        """
        주어진 픽셀에서 K-Means를 사용하여 지배적인 색상을 추출하고 클러스터 크기(픽셀 수) 기준으로 정렬합니다.
        """
        if len(pixels) == 0:
            return None
        
        if len(pixels) < n_colors:
            n_colors = len(pixels)
            if n_colors == 0:
                return None

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 클러스터별 픽셀 수 계산
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # (픽셀 수, 클러스터 중심 색상) 쌍을 만든 후 픽셀 수 기준으로 내림차순 정렬
        dominant_colors_sorted_by_count = sorted(
            zip(counts, kmeans.cluster_centers_), 
            key=lambda x: x[0], 
            reverse=True
        )
        
        # 정렬된 색상만 추출
        sorted_colors = [color.astype(int) for count, color in dominant_colors_sorted_by_count]

        return sorted_colors

    def rgb_to_hex(self, rgb: np.ndarray) -> str:
        """RGB를 HEX 코드로 변환"""
        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

    def extract_face_colors(self, image: Image.Image) -> (Dict[str, List[str]], np.ndarray, np.ndarray):
        """얼굴 부위별 색상 HEX 코드만 추출하는 메인 함수"""
        original_image, segmentation_mask = self.parse_face_from_memory(image)
        
        part_hex_codes = {}
        for part_name in self.target_parts.keys():
            part_hex_codes[part_name] = []

        for part_name, label_ids in self.target_parts.items():
            part_mask = np.isin(segmentation_mask, label_ids)
            if np.any(part_mask):
                part_pixels = original_image[part_mask]

                # 'eyes'인 경우 흰색/회색 영역 제외 (검은색은 포함)
                if part_name == "eyes":
                    hsv_pixels = cv2.cvtColor(part_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                    # 흰색/회색 제외: 채도(S)가 낮고 명도(V)가 높은 픽셀을 제외
                    non_white_mask = ~((hsv_pixels[:, 1] < 30) & (hsv_pixels[:, 2] > 180))
                    part_pixels = part_pixels[non_white_mask]

                    if len(part_pixels) == 0:
                        continue # 흰색/회색 제외 후 픽셀이 없으면 다음 부위로
                
                n_colors = 3 # K-means로 3개의 색상 추출
                
                colors_list = self._extract_sorted_dominant_colors(
                    part_pixels, n_colors=n_colors
                )

                if colors_list is not None:
                    # 추출된 3개의 색상 중 상위 2개만 사용
                    part_hex_codes[part_name] = [self.rgb_to_hex(c) for c in colors_list[:2]]
        
        return part_hex_codes, original_image, segmentation_mask

    def _plot_color_palette(self, axes, colors: Dict[str, List[str]]):
        """HEX 코드만 사용하여 색상 팔레트 시각화"""
        axes.axis('off')
        axes.set_title("Extracted Colors", fontsize=12, weight='bold')

        y_pos = 0.8
        
        for part_name, hex_codes in colors.items():
            if not hex_codes:
                continue
            
            axes.text(0.05, y_pos, f"{part_name.title()}:", fontsize=10, va='center', weight='bold')
            
            for i, hex_code in enumerate(hex_codes):
                axes.add_patch(plt.Rectangle(((0.3 + i * 0.2), y_pos - 0.04), 0.15, 0.08, 
                                          facecolor=hex_code, edgecolor='black'))
                axes.text(0.375 + i * 0.2, y_pos - 0.07, hex_code, fontsize=8, ha='center')
            
            y_pos -= 0.2
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

    def _plot_part_masks(self, axes, original_image, segmentation_mask):
        """부위별 마스크 시각화"""
        part_names = list(self.target_parts.keys())
        for i, part_name in enumerate(part_names[:3]):
            ax = axes[1, i]
            label_ids = self.target_parts[part_name]
            part_mask = np.isin(segmentation_mask, label_ids)
            
            ax.imshow(original_image)
            if np.any(part_mask):
                colored_mask = np.zeros_like(original_image)
                colored_mask[part_mask] = [255, 0, 0] # Red
                ax.imshow(colored_mask, alpha=0.5)
            
            ax.set_title(f"{part_name.title()} Mask")
            ax.axis('off')

    def visualize_results(self, original_image, segmentation_mask, colors, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(segmentation_mask, cmap='tab20')
        axes[0, 1].set_title("Segmentation Mask")
        axes[0, 1].axis('off')
        
        self._plot_color_palette(axes[0, 2], colors)
        self._plot_part_masks(axes, original_image, segmentation_mask)

        axes[1, 2].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"분석 결과 이미지를 '{save_path}'에 저장했습니다.")

async def main(file: UploadFile):
    """얼굴 이미지에서 HEX 코드만 추출하고 시각화하는 메인 함수"""
    extractor = FaceColorExtractor()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        hex_codes_data, original_image, segmentation_mask = extractor.extract_face_colors(image)
        
        extractor.visualize_results(original_image, segmentation_mask, hex_codes_data, "face_color_analysis.png")
        
        return hex_codes_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

async def extract_face_only(file: UploadFile):
    """얼굴만 추출하고 배경을 제거하는 함수"""
    extractor = FaceColorExtractor()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        original_image, segmentation_mask = extractor.parse_face_from_memory(image)
        
        face_labels = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        face_mask = np.isin(segmentation_mask, face_labels)
        
        if not np.any(face_mask):
            raise HTTPException(status_code=400, detail="얼굴 영역을 찾을 수 없습니다.")
        
        face_only_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        face_only_image[:, :, :3] = original_image
        face_only_image[:, :, 3] = face_mask.astype(np.uint8) * 255
        
        face_coords = np.where(face_mask)
        y_min, y_max = np.min(face_coords[0]), np.max(face_coords[0])
        x_min, x_max = np.min(face_coords[1]), np.max(face_coords[1])
        
        cropped_face = face_only_image[y_min:y_max+1, x_min:x_max+1]
        cropped_pil = Image.fromarray(cropped_face, 'RGBA')
        
        img_byte_arr = io.BytesIO()
        cropped_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"얼굴 추출 중 예상치 못한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")