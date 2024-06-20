import math
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import ImageProcessingMixin, ProcessorMixin, SiglipImageProcessor, AutoTokenizer, AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class MultiCropImageProcessor(ImageProcessingMixin):
    def __init__(self, model_name, max_crops=0, **kwargs):
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.crop_size = 384
        self.max_crops = max_crops
        self.stride_ratio = 2

    def __call__(
        self,
        images: List[Image.Image],
        max_crops: int = -1,
    ):
        res = {
            "pixel_values": [],
            "coords": [],
        }
        if max_crops < 0:
            max_crops = self.max_crops
        for image in images:
            outputs, output_coords = self.process_image(image, max_crops)
            res["pixel_values"].append(outputs)
            res["coords"].append(output_coords)
            return res

    def process_image(
        self,
        image: Image.Image,
        max_crops: int
    ):
        outputs = []
        output_coords = []
        outputs.append(self.processor(image, return_tensors="pt").pixel_values)
        output_coords.append(torch.tensor([0.5, 0.5]))
        width, height = image.size
        crop_size = self.crop_size
        stride = crop_size // self.stride_ratio
        if (
            max_crops == 0
            or width <= (crop_size + stride)
            and height <= (crop_size + stride)
        ):
            outputs = torch.cat(outputs, dim=0)
            output_coords = torch.cat(output_coords, dim=0)
            return outputs, output_coords
        total_tokens = math.inf
        while total_tokens > max_crops:
            total_tokens = (
                math.floor((width - crop_size) / stride) + 1
            ) * (
                math.floor((height - crop_size) / stride) + 1
            )
            if total_tokens > max_crops:
                crop_size += 10
                stride = crop_size // self.stride_ratio
        stride = crop_size // self.stride_ratio
        x_steps = int(math.floor((width - crop_size) / stride) + 1)
        if x_steps < 1:
            x_steps = 1
        y_steps = int(math.floor((height - crop_size) / stride) + 1)
        if y_steps < 1:
            y_steps = 1
        if x_steps == 1 and y_steps == 1:
            outputs = torch.cat(outputs, dim=0)
            output_coords = torch.cat(output_coords, dim=0)
            return outputs, output_coords
        x_coords = []
        y_coords = []
        for i in range(x_steps):
            x_coords.append([i * stride, i * stride + crop_size])
        if x_coords[-1][1] != width:
            x_coords[-1][1] = width
        for i in range(y_steps):
            y_coords.append([i * stride, i * stride + crop_size])
        if y_coords[-1][1] != height:
            y_coords[-1][1] = height
        image_parts = []
        part_coords = []
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                image_parts.append(
                    image.crop(
                        (x_coords[i][0], y_coords[j][0], x_coords[i][1], y_coords[j][1])
                    )
                )
                part_coords.append(
                    torch.tensor(
                        [
                            (x_coords[i][0] + x_coords[i][1]) / 2 / width,
                            (y_coords[j][0] + y_coords[j][1]) / 2 / height,
                        ]
                    )
                )
        for image_part in image_parts:
            outputs.append(self.processor(image_part, return_tensors="pt").pixel_values)
        for part_coord in part_coords:
            output_coords.append(part_coord)
        outputs = torch.cat(outputs, dim=0)
        output_coords = torch.stack(output_coords, dim=0)
        return outputs, output_coords


class LlavaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = MultiCropImageProcessor
    tokenizer_class = "SiglipTokenizer"

    def __init__(self, image_processor: MultiCropImageProcessor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.search_model = None
    
    @classmethod
    def from_pretrained(cls, path, trust_remote_code=True, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        image_processor = MultiCropImageProcessor(path, trust_remote_code=trust_remote_code)
        return LlavaProcessor(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        images: ImageInput = None,
        model = None,
        max_crops: int = 0,
        num_tokens = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if images is not None:
            processor_outputs = self.image_processor(images, max_crops)
            pixel_values = processor_outputs["pixel_values"]
            pixel_values = [
                value.to(model.device).to(model.dtype) for value in pixel_values
            ]
            coords = processor_outputs["coords"]
            coords = [value.to(model.device).to(model.dtype) for value in coords]
            image_outputs = model.vision_model(pixel_values, coords, num_tokens)
            image_features = model.multi_modal_projector(image_outputs)
        else:
            image_features = None
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        text_inputs['input_ids'] = text_inputs['input_ids'].to(model.device)
        text_inputs['attention_mask'] = text_inputs['attention_mask'].to(model.device)
        return BatchFeature(data={**text_inputs, "image_features": image_features})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
