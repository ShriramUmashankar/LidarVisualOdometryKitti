from VisualOdometry import MotionEstimation


dataSet = MotionEstimation('00')
VisualOdometry = dataSet.motion_estimator(SaveData = True, PlotData = False)