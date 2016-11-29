from img_loading import ImageLoader
from feature import pc_analyzer

train_data = ImageLoader().getData(5)
means = pc_analyzer.calculate_mean(train_data)