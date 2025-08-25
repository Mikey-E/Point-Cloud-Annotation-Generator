import base64
import argparse
import os
import mimetypes
import sys
from openai import OpenAI

client = OpenAI()

def encode_image(image_path: str) -> str:
    """Read an image from disk and return a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def detect_mime(image_path: str) -> str:
    """Best-effort mime type detection; default to image/jpeg if unknown."""
    mt, _ = mimetypes.guess_type(image_path)
    if mt is None:
        return "image/jpeg"
    return mt

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send an image to the OpenAI vision model and get a caption/answer.")
    p.add_argument("image", help="Path to the image file")
    p.add_argument("--prompt", default="what's in this image?", help="Prompt/question to ask about the image")
    # p.add_argument("--model", default="gpt-4.1", help="Model ID to use")
    p.add_argument("--model", default="gpt-5-nano", help="Model ID to use")
    p.add_argument("--no-strip", action="store_true", help="Do not strip leading/trailing whitespace from the response")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image):
        parser.error(f"Image path not found: {args.image}")

    mime = detect_mime(args.image)
    b64 = encode_image(args.image)

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )

    out = completion.choices[0].message.content
    if not args.no_strip:
        out = out.strip()
    print(out)

if __name__ == "__main__":
    main()