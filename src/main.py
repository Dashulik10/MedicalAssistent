from extraction.image_processor import DataExtractorAgent
from pathlib import Path
from time import perf_counter

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGE_DIR = DATA_DIR / "good_images"
IMAGE_LIST = list(IMAGE_DIR.glob("*.png"))

start = perf_counter()

agent = DataExtractorAgent()
random_image = IMAGE_LIST[0]
print(random_image)
data = agent.extract_data(random_image)
print(data)

end = perf_counter()
print(f"Time taken: {end - start}")
