"""
Author: rciszek
"""
import cv2
import numpy as np

class VideoAnalyzer:
    """
        Performs analyses on video frames.
        
        Arguments:
            movement_threshold:             Threshold for the ratio of changed pixels between 
                                            frames. If the ratio is exceeded, the frame is
                                            concluded to contain movement.
            open_kernel_size:               Size of the opening kernel used for removing small 
                                            changes such as noise from the frame.
            history:                        The number of past frames included in the analysis.
            mixtures:                       The number of mixtures used for foreground segmentation.
            background_ratio:               Background ratio parameter for foreground segmentation.
            complexity_reduction_threshold: Complexity reduction threshold for foreground segmentation.       
    """
    
    def __init__(self, movement_threshold=0.001, open_kernel_size=3, history = 100, mixtures=5, background_ratio = 0.8, complexity_reduction_threshold=0.05):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.movement_threshold = movement_threshold
        self.history = history
        self.mixtures = mixtures
        self.background_ratio = background_ratio
        self.complexity_reduction_threshold = complexity_reduction_threshold
        self.open_kernel_size = open_kernel_size
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_kernel_size,open_kernel_size))
        
    def updateParameters(self):
        """
        Updates the foreground separator to take account the current settings.
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgbg.setHistory(int(self.history))
        self.fgbg.setNMixtures(int(self.mixtures))
        self.fgbg.setBackgroundRatio(float(self.background_ratio))
        self.fgbg.setComplexityReductionThreshold(float(self.complexity_reduction_threshold))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(self.open_kernel_size),int(self.open_kernel_size)))
        
    def getParameters(self):
        """
        Return the current parameters as a dict.
        """        
        return dict( movement_threshold = self.movement_threshold, history = self.mixtures, mixtures = self.mixtures, background_ratio = self.background_ratio, open_kernel_size = self.open_kernel_size )
        
    def detectMovement(self, frame):
        """
        Detects movement from the given frame.
        
        Arguments:
            frame: Videoframe
            
        Outputs:
            foreground_mask: The input videoframe masked to highlight the 
                             foreground object.
            movement:        Boolean value indicating the presence of movement.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        foreground_mask = self.fgbg.apply(frame,learningRate=-1)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, self.open_kernel)
        
        movement = False
        
        if self.movement_threshold < float(np.count_nonzero(foreground_mask)) / float((foreground_mask.shape[0]*foreground_mask.shape[1])):
            movement = True
            
        return foreground_mask, movement

class MovementTracker:
    """
    Encapsulates the tracking of movements.
    """
    def __init__(self):
        self.events = []
        self.previously_moving = False
        self.startTime = 0
        
    def update(self, currently_moving, time):
        """
        Updates the current movement state
        
        Arguments:
            currently_moving:   Boolean value indicating current movement state.
            time:               Current time  
        """
        
        if currently_moving == True and self.previously_moving == False:
            self.previously_moving = True
            self.startTime = time          
        if currently_moving == False and self.previously_moving == True:         
            self.events.append([self.startTime, time])
            self.previously_moving = False
            self.startTime = 0         
            
    def getEvents(self):
        """
        Returns the event array
        """
        return np.array(self.events)
