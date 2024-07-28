from stlddec.decomposition import TaskOptiContainer, powerset
from stlddec.decomposition import TimeInterval,TemporalOperator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ax = plt.gca()
# Display the background image, ensuring it covers the axis limits
# Load the image using Matplotlib
background_image = mpimg.imread("/home/gregorio/Desktop/papers/journal/decentralized_STL_task_decomposition/code/assets/pia21451-ezgif.com-webp-to-jpg-converter.jpg")
ax.imshow(background_image, extent=[ax.get_xlim()[0],ax.get_xlim()[1], ax.get_ylim()[0],ax.get_ylim()[1]], aspect='auto')
plt.show()