import numpy as np
from nilearn.image import new_img_like

class RandomRegionGrower:
    ''''A class to grow random binary regions'''
    
    def __init__(self,mask_img,rng):
        
        self.mask_img = mask_img
        self.mask_data = np.asarray(self.mask_img.dataobj)
        self.rng = rng
        self.region_data = np.zeros_like(self.mask_data)
    
    def get_random_seed(self):
        
        indices_where_true = np.argwhere((self.mask_data == 1) & (self.region_data == 0))
        random_int = self.rng.randint(len(indices_where_true),size=1)
        random_seed = indices_where_true[random_int][0]
        random_seed = (random_seed[0],random_seed[1],random_seed[2])
        
        return random_seed
    
    def grow(self,region_sizes):
        
        for region_size in region_sizes:
        
            self.region_size = region_size
            self.new_voxel = self.get_random_seed()
            self.region_data[self.new_voxel[0],self.new_voxel[1],self.new_voxel[2]] = 1
            self.queue = list()
            self.queue.append((self.new_voxel[0],self.new_voxel[1],self.new_voxel[2]))
            self.region_size_current = 1
                        
            while len(self.queue) != 0:
                
                # we pop a random voxel out of the queue (this is necessary
                # otherwise we get a systematic grow in one direction)
                self.new_voxel = self.queue.pop(self.rng.randint(len(self.queue),size=1)[0])
                
                # check all six neighbours of current voxel
                self.check_neighbour(self.new_voxel[0]-1,self.new_voxel[1],  self.new_voxel[2])
                self.check_neighbour(self.new_voxel[0]+1,self.new_voxel[1],  self.new_voxel[2])
                self.check_neighbour(self.new_voxel[0],  self.new_voxel[1]-1,self.new_voxel[2])
                self.check_neighbour(self.new_voxel[0],  self.new_voxel[1]+1,self.new_voxel[2])
                self.check_neighbour(self.new_voxel[0],  self.new_voxel[1],  self.new_voxel[2]-1)
                self.check_neighbour(self.new_voxel[0],  self.new_voxel[1],  self.new_voxel[2]+1)
            
            self.other_regions = self.region_data
    
        return new_img_like(self.mask_img,self.region_data)
        
    def check_neighbour(self,x,y,z):
        if self.mask_data[x,y,z] == 1 and self.region_data[x,y,z] == 0 and self.region_size_current < self.region_size:
            self.region_data[x,y,z] = 1
            self.queue.append((x,y,z))
            self.region_size_current += 1
    
if __name__ == '__main__':
    pass