import cv2
from PIL import Image
import numpy as np

def screenshot_to_publication(screenshot_path, output_path, target_width, ppi=300):
  """
  Transforms a screenshot into a publication-standard image.

  Args:
    screenshot_path: Path to the screenshot image.
    output_path: Path to save the processed image.
    target_width: Desired width of the final image in pixels.
    ppi: Pixels per inch resolution (default 300 for print).

  Returns:
    None
  """

  # Read the screenshot
  img = cv2.imread(screenshot_path)

  # Crop unnecessary elements (modify as needed)
  # img = img[100:500, 200:700]  # Example crop

  # Resize proportionally to target width
  height = int(img.shape[0] * target_width / img.shape[1])
  img = cv2.resize(img, (target_width, height), interpolation=cv2.INTER_AREA)

  # Convert to sRGB color space
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Adjust contrast and brightness (modify as needed)
  img = cv2.addWeighted(img, 1.2, img, 0, -20)

  # Reduce noise (adjust strength as needed)
  img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

  # Sharpen (adjust strength as needed)
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  img = cv2.filter2D(img, -1, kernel)

  # Convert to PIL Image and save
  image = Image.fromarray(img)
  image.save(output_path, quality=95)  # Adjust quality as needed

# Example usage
screenshot_path = "./Screenshot (142)_1.png"
output_path = "publication_ready_SS(142)_1.jpg"
target_width = 1000  # Adjust as needed

screenshot_to_publication(screenshot_path, output_path, target_width)

print(f"Publication-ready image saved to: {output_path}")