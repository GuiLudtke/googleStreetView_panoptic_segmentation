import sys
import os
from streetview_downloader import import_coordinates,get_streetview_images
from panoptic_segmentation2 import generate_results

# -------------------- Execução principal via terminal --------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python run_segmentation.py <caminho_do_arquivo_api_key>")
        sys.exit(1)

    caminho_api_key = sys.argv[1]

    with open(caminho_api_key, 'r') as f:
        api_key = f.read().strip()
    
    os.makedirs("results", exist_ok=True)
    coords_dict = import_coordinates("data/coordinates")
    get_streetview_images(coords_dict, api_key)
    generate_results("data/streetView_images")

    