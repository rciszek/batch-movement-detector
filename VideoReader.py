
import numpy as np
import subprocess as sp
from deployment import resource_path
import os

class VideoReader:
    
    DEFAULT_FPS = 30
    FFMPEG_BIN_WIN = "ffmpeg/ffmpeg.exe"
    FFPROBE_BIN_WIN = "ffmpeg/ffprobe.exe"
    NO_CONSOLE_FLAG = 0x08000000
    
    def __init__(self,file_path):
        self.file_path = file_path
        self.width, self.height, self.fps, self.duration, self.frames = self.getVideoProperties(file_path)  
        self.currentPositionInFrames = 0
        self.openPipe(file_path,0)
        
        
                
    def openPipe(self, file_name,position):
        
        command = [ resource_path(VideoReader.FFMPEG_BIN_WIN),
                '-ss', str(position),                   
                '-i', resource_path(file_name),
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']   
        self.pipe = sp.Popen(command, stdout = sp.PIPE, creationflags  = VideoReader.NO_CONSOLE_FLAG  )

    def getVideoProperties(self,file_path):  

        command = [ resource_path(VideoReader.FFPROBE_BIN_WIN),
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               file_path]
        ffprobe = sp.Popen(command, stdout = sp.PIPE, creationflags  = VideoReader.NO_CONSOLE_FLAG )
        out, error = ffprobe.communicate()
        ffprobe.stdout.close()
        ffprobe.kill()
        width = 0
        height = 0
        fps = 0
        duration = 0
        frames = 0
        out = out.decode("utf-8") 
        out = out.split('\r\n')
        if not error:
            width = int(out[0])
            height = int(out[1])
            fps = float(out[2].split('/')[0])/float(out[2].split('/')[1])
            duration = float(out[3])
            frames = int(out[4])

                
        return width, height, fps, duration, frames
        
    def nextFrame(self):

        if self.pipe.stdout.closed:
            return None
        
        raw_image = self.pipe.stdout.read(self.width*self.height*3)
        frame =  np.fromstring(raw_image, dtype='uint8')
        if frame.shape[0] == 0:
            return None
        frame = frame.reshape((self.height,self.width,3))

        self.currentPositionInFrames += 1
        
        return frame
        
    
    def currentPositionInSeconds(self):
        return float(self.currentPositionInFrames / self.fps)
    
    def lengthInSeconds(self):
        return self.duration
    
    
    def setPositionInSeconds(self, position):
        self.currentPositionInFrames = position*self.fps 
        self.close()
        self.openPipe(self.file_path,position)
    
    def close(self):
        self.pipe.stdout.close()
        self.pipe.kill()
        
    def isClosed(self):
        if self.pipe.poll() is None:
            return True
        else:
            return False
        
    def getFrameData(self, frame):
        height = 0
        width = 0
        channel = 0
        bytesPerLine = 0
        
        if frame is None:
             return height, width, channel, 0
        
        if frame.ndim == 3:
            height, width, channel = frame.shape
        if frame.ndim == 2:
            height, width = frame.shape
        bytesPerLine = 3 * width 
        
        return height, width, channel, bytesPerLine
