from feature.pca import PCA
from feature.Interests import Interest_points
from feature import MomentsCalculator

NUM_EVECTS_PCA = 2
pc_analyzer = PCA(NUM_EVECTS_PCA)
moment_calc = MomentsCalculator()
interest_pt_analyzer = Interest_points()