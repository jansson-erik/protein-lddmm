import numpy as np
import mdtraj as md 
from scipy.linalg import orthogonal_procrustes
from concurrent.futures import ThreadPoolExecutor



def procrustes(data1, data2):   
  mtx1 = np.array(data1, dtype=np.double, copy=True)
  mtx2 = np.array(data2, dtype=np.double, copy=True) 
  mtx2 -= np.mean(mtx2, 0)-np.mean(mtx1, 0)
  norm1 = np.linalg.norm(mtx1)
  norm2 = np.linalg.norm(mtx2)
  mtx2 /= norm2/norm1
  R, s = orthogonal_procrustes(mtx1, mtx2)
  mtx2 = np.dot(mtx2, R.T) #* s
  disparity = np.sum(np.square(mtx1 - mtx2))
  return mtx1, mtx2, disparity

class Data:
    """
    Data class.
    """
    def __init__(self, N, grid,forward_model, manyproj = True,blob_size=0.5, noise_sigma=0.1):
        """
        Constructor function. 
        Parameters
        ----------
        N: int, number of pixels.
        grid: object of grid class.
        forward_model: object of forward class.
        manyproj: bool, whether to use many projections. Default = True.
        blob_size: float, size of the blob. Default = 0.5.
        noise_sigma: float, standard deviation of the noise. Default = 0.1.
        """
        self.manyproj = manyproj
        self.N = N
        self.grid = grid
        self.blob_size = blob_size
        self.noise_sigma=noise_sigma
 
        self.forward_model = forward_model
        if manyproj:
            Data.real_chain_Nproj(self)
        else:
            Data.real_chain_xyz(self)
        
    
        
    def select_segment(positions, frame, i0, i1):
        """
        Returns a subset of the positions of a given frame.  
        
        Parameters
        ----------
        positions: np.array, shape = (n_frames,n_atoms,3)
        frame: int
        i0: int
        i1: int

        Returns
        -------
        positions: np.array, shape = (n_atoms,3).
        """
        return positions[frame, i0:i1]
    def select_positions(filename,top =None ):
        """
        This functions returns the positions of the CA atoms in a given backbone. 

        Parameters
        ----------
        filename: str
        top: str optional, default = None. 
        """
        if top is None:
            trajectory = md.load(filename)
        else:
            trajectory = md.load(filename,top= top)
        indices_CA = []
        indices_back = []
        for atom in trajectory.topology.atoms:
            if atom.name in ['N','CA','C']:
                indices_back.append(atom.index)
            if atom.name in ['CA']:
                indices_CA.append(atom.index)
        CA_pos = trajectory.xyz[:,indices_CA,:]*10.0
        return CA_pos
    
    def real_chain_Nproj(self):
        """
        This functions adds the target and template protein conformations, their projections, the blurred images and the noisy target images. 

        Parameters
        ----------

        Returns
        -------

        """
        CA_pos = Data.select_positions('data/struct1/dims0001_fit-core.dcd',top = 'data/struct1/adk4ake.psf')
        
        target_pc = Data.select_segment(CA_pos,-1,0,self.N+1)
        
        target_pc = target_pc - np.mean(target_pc,axis = 0)
        
        template_pc_md = Data.select_segment(CA_pos,0,0,self.N+1)
        
       
        template_pc_md = template_pc_md -np.mean(template_pc_md,axis = 0)
        
    

        self.template_model_md = self.forward_model.invM(template_pc_md) #- np.mean(template_model,axis = 0)
       
        CA_pos = Data.select_positions('/Users/erikjans/Downloads/AF-P69441-F1-model_v4.pdb')
        
            
        template_pc = Data.select_segment(CA_pos,0,0,self.N+1)
        template_pc = procrustes(template_pc_md,template_pc)[1]
        
        target_model = self.forward_model.invM(target_pc)
        template_model = self.forward_model.invM(template_pc)
        
        self.template_model = template_model
        self.target_model = target_model
        self.target_pc = target_pc
        self.template_model_md = self.forward_model.invM(template_pc_md) #- np.mean(template_model,axis = 0)
        
        with ThreadPoolExecutor() as executor:
            self.target_pc_projections = list(executor.map(lambda proj: self.forward_model.project(target_pc, proj), self.forward_model.projections))
            self.template_pc_projections = list(executor.map(lambda proj: self.forward_model.project(template_pc, proj), self.forward_model.projections))
            self.target_projections_blur_nb = list(executor.map(lambda proj: self.forward_model.blur(proj, self.grid, self.blob_size), self.target_pc_projections))
            self.template_projections_blur = list(executor.map(lambda proj: self.forward_model.blur(proj, self.grid, self.blob_size), self.template_pc_projections))
            self.target_projections_blur = [image + self.noise_sigma * np.random.randn(*image.shape) for image in self.target_projections_blur_nb]

        
    def real_chain_xyz(self):
        """
        Old function, to be removed after testing. This function returns the CA-positions of the target and template models. 
        """
        CA_pos = Data.select_positions('data/struct1/dims0001_fit-core.dcd',top = 'data/struct1/adk4ake.psf')

        target_pc = Data.select_segment(CA_pos,-1,0,self.N+1)
        target_pc = target_pc - np.mean(target_pc,axis = 0)
        
        
        
        template_pc_md = Data.select_segment(CA_pos,0,0,self.N+1)
        template_pc_md = template_pc_md -np.mean(template_pc_md,axis = 0)
        
        template_pc = Data.select_segment(CA_pos,0,0,self.N+1)
        template_pc = procrustes(template_pc_md,template_pc)[1]

        target_model = self.forward_model.invM(target_pc)
        template_model = self.forward_model.invM(template_pc)

         
        trajectory = md.load('/Users/erikjans/Downloads/AF-P69441-F1-model_v4.pdb')
                ## We pick out the positions of the whole backbone and the positions of just the CA-atoms.
        indices_CA = []
        indices_back = []
        for atom in trajectory.topology.atoms:
            if atom.name in ['N','CA','C']:
                indices_back.append(atom.index)
            if atom.name in ['CA']:
                indices_CA.append(atom.index)
         # We have to multiply the trajectory by 10 so that the interatomic distance is in Angstroms
        CA_pos = trajectory.xyz[:,indices_CA,:]*10.0#+ 0.5#10
       
        
        template_pc = Data.select_segment(CA_pos,0,0,self.N+1)
        template_pc = procrustes(template_pc_md,template_pc)[1]

        
        
        
        target_model = self.forward_model.invM(target_pc)
        template_model = self.forward_model.invM(template_pc)

        target_pcz = self.forward_model.Pz(target_pc)
        template_pcz = self.forward_model.Pz(template_pc)

        target_pcx = self.forward_model.Px(target_pc) 
        template_pcx = self.forward_model.Px(template_pc)


        template_pcy = self.forward_model.Py(template_pc)
        
        target_pcy = self.forward_model.Py(target_pc)
        
        target_blur_y = self.forward_model.blur(target_pcy, self.grid, self.blob_size)
        template_blur_y = self.forward_model.blur(template_pcy, self.grid, self.blob_size)
        
        target_blur_z = self.forward_model.blur(target_pcz, self.grid, self.blob_size)
        template_blur_z = self.forward_model.blur(template_pcz, self.grid, self.blob_size)
        
        target_blur_x = self.forward_model.blur(target_pcx, self.grid, self.blob_size)
        template_blur_x = self.forward_model.blur(template_pcx, self.grid, self.blob_size)
        
        self.target_nb_x = target_blur_x
        self.target_nb_y = target_blur_y
        self.target_nb_z = target_blur_z
        target_blur_z = target_blur_z + self.noise_sigma*np.random.randn(*target_blur_z.shape)
        target_blur_x = target_blur_x +self.noise_sigma*np.random.randn(*target_blur_x.shape)
        target_blur_y = target_blur_y + self.noise_sigma*np.random.randn(*target_blur_y.shape)
        
        
        template_model = self.forward_model.invM(template_pc)
        self.template_model = template_model
        
       
        self.target_model = target_model

        self.target_pc = target_pc
        
        self.target_pcz = target_pcz
        self.target_pcx = target_pcx
        self.target_pcy = target_pcy 
        
        self.target_blur_z = target_blur_z
        self.target_blur_x = target_blur_x
        self.target_blur_y = target_blur_y 

        self.template_pc = template_pc
        self.template_pcz = template_pcz
        self.template_pcx = template_pcx
        self.template_pcy = template_pcy 
        
        self.template_blur_z = template_blur_z
        self.template_blur_x = template_blur_x
        self.template_blur_y = template_blur_y


