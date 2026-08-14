import os
from pathlib import Path
from docling.document_converter import DocumentConverter

"""
Torch compiler includes OpenAI Triton - which is the default compiler
library backend used by torch.compile. The 5060 gpu was too new for
the standard execution code which resulted it in invalid device errors.
However, it proves to be a minor bottleneck which I can move ahead without.
For users with similar issues, I hear:
compiled_model = torch.compile(model, backend="aot_eager")
Maybe of some help.

With enabling my gpu it took me around 6 minuites whereas it took me
22 minuites for the cpu. Interesting. USE THE GPU!
"""
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Folders
INPUT_DIR = Path("eLibrarySansadDownloads")
OUTPUT_DIR = Path("FormattedExtraction")

# Create output folder if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Load models only once
converter = DocumentConverter()

pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF(s).\n")

for i, pdf_path in enumerate(pdf_files, start=1):
    print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")

    try:
        result = converter.convert(pdf_path)

        markdown = result.document.export_to_markdown()

        output_file = OUTPUT_DIR / f"{pdf_path.stem}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"✓ Saved: {output_file.name}")

    except Exception as e:
        print(f"✗ Failed: {pdf_path.name}")
        print(e)

print("\nDone!")