"""
fabric_style_transfer.py
------------------------
Refactored OOP-based fabric style transfer script using the official Google GenAI SDK.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Official SDK and Image processing
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fabric_transfer")


# ---------------------------------------------------------------------------
# Enums & Configuration
# ---------------------------------------------------------------------------

class FabricType(str, Enum):
    """Known fabric hint tokens."""
    DENIM         = "denim"
    LINEN         = "linen"
    SILK          = "silk"
    VELVET        = "velvet"
    TWEED         = "tweed"
    LEATHER       = "leather"
    COTTON_CANVAS = "cotton canvas"
    KNIT_WOOL     = "knit wool"
    CUSTOM        = "custom"


class AnchorModifier(str, Enum):
    """Prompt tokens to guide the AI's behavior."""
    KEEP_BACKGROUND   = "keep background unchanged"
    PRESERVE_SHAPE    = "preserve shape and silhouette"
    PHOTOREALISTIC    = "photorealistic"
    MATCH_LIGHTING    = "match original lighting and shadows"
    WEAVE_DETAIL      = "visible weave and thread detail"


DEFAULT_ANCHORS: frozenset[AnchorModifier] = frozenset({
    AnchorModifier.KEEP_BACKGROUND,
    AnchorModifier.PRESERVE_SHAPE,
    AnchorModifier.PHOTOREALISTIC,
    AnchorModifier.MATCH_LIGHTING,
})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TransferRequest:
    """Everything needed for one style transfer job."""
    input_image_path: str | Path
    style_image_path: str | Path
    target_object: str
    fabric_type: FabricType = FabricType.CUSTOM
    fabric_hint: str = ""
    anchors: frozenset[AnchorModifier] = field(default_factory=lambda: DEFAULT_ANCHORS)
    output_path: str | Path = "output.png"
    # Note: Nano Banana 2 automatically handles complexity; 
    # specific 'steps' or 'guidance' are handled via the SDK config.


@dataclass
class TransferResult:
    """Result of a completed job."""
    success: bool
    output_path: str | Path
    prompt_used: str
    error: str = ""


# ---------------------------------------------------------------------------
# Logic Components
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Assembles the prompt for the Gemini Image Edit model."""

    QUALITY_SUFFIX = "8k resolution, high-quality fabric photography, sharp focus."

    def build(self, req: TransferRequest) -> str:
        fabric_str = self._resolve_fabric(req)
        anchor_str = ", ".join(a.value for a in sorted(req.anchors, key=lambda x: x.value))
        
        parts = [
            f"Apply the fabric style from the reference image onto {req.target_object}.",
            f"The object should look like it is made of {fabric_str}.",
            anchor_str,
            self.QUALITY_SUFFIX,
        ]
        return " ".join(filter(None, parts))

    @staticmethod
    def _resolve_fabric(req: TransferRequest) -> str:
        if req.fabric_type == FabricType.CUSTOM:
            return req.fabric_hint or "the fabric shown in the reference image"
        return req.fabric_type.value


class GeminiStyleClient:
    """Wrapper for the official Google GenAI SDK."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, req: TransferRequest) -> bytes:
        """Uses the image_edit tool (Nano Banana 2) to perform the transfer."""
        
        # Open images as PIL objects for the SDK
        base_img = Image.open(req.input_image_path)
        style_ref = Image.open(req.style_image_path)

        log.info("Sending request to Gemini API (Nano Banana 2)...")
        
        # The edit_image call takes the base image and a list of reference images
        response = self.client.models.edit_image(
            model="gemini-2.0-flash",
            prompt=prompt,
            base_image=base_img,
            reference_images=[style_ref]
        )

        if not response.generated_images:
            raise RuntimeError("API returned no images.")

        return response.generated_images[0].image_bytes


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class StyleTransferPipeline:
    def __init__(self, api_key: str, dry_run: bool = False):
        self.builder = PromptBuilder()
        self.client = GeminiStyleClient(api_key) if not dry_run else None
        self.dry_run = dry_run

    def run(self, req: TransferRequest) -> TransferResult:
        log.info("=== Style Transfer Job Started ===")
        prompt = self.builder.build(req)
        log.info(f"Generated Prompt: {prompt}")

        if self.dry_run:
            log.info("[Dry-run] Skipping API call.")
            return TransferResult(True, "(dry-run)", prompt)

        t0 = time.time()
        try:
            image_bytes = self.client.generate(prompt, req)
            
            output_path = Path(req.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            elapsed = time.time() - t0
            log.info(f"Job completed in {elapsed:.2f}s. Saved to {output_path}")
            return TransferResult(True, output_path, prompt)

        except Exception as e:
            log.error(f"Error during transfer: {e}")
            return TransferResult(False, "", prompt, error=str(e))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fabric Style Transfer via Gemini API")
    parser.add_argument("--input", required=True, help="Path to object image")
    parser.add_argument("--style", required=True, help="Path to fabric sample image")
    parser.add_argument("--object", required=True, help="Name of object (e.g., 'the sofa')")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--fabric", default="custom", choices=[f.value for f in FabricType])
    parser.add_argument("--fabric-hint", default="")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        print("Error: API Key required. Set GEMINI_API_KEY environment variable or use --api-key.")
        return

    #genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    req = TransferRequest(
        input_image_path=args.input,
        style_image_path=args.style,
        target_object=args.object,
        fabric_type=FabricType(args.fabric),
        fabric_hint=args.fabric_hint,
        output_path=args.output
    )

    pipeline = StyleTransferPipeline(api_key=args.api_key, dry_run=args.dry_run)
    pipeline.run(req)


if __name__ == "__main__":
    main()

#SAMPLE CALL
#python fabricTransfer.py --input couch.jpg --style fabric.jpg --object "the sofa"

#REQUIRES AN API KEY
#ALSO REQUIRES API_KEY TO BE SET as an ENV variable


#### PROMPT USED TO QUERY ####
'''
Create a professional fashion product photo. 
Take the jeans from the jeans.jpeg and apply the exact fabric texture, 
color, and weave pattern  from fabric.jpeg onto it. 

Requirements:
- The {target_object} must maintain its original shape, folds, and silhouette.
- Match the lighting and shadows of the original scene for a photorealistic look.
- Keep the background exactly as it is in the first image.
- High-resolution, 8k, sharp focus on fabric detail.
keep background unchanged,preserve shape and silhouette,photorealistic,match original lighting and shadows,visible weave and thread detail
'''