import cv2
from vmbpy import *
import vmbpy
import functions
from functions import *
import numpy as np
import os



freq = 3                        # initial freq for projected fringes
displacement, resize_h, resize_w, = monitor_info()
base_path = os.path.dirname(__file__) # Directorio para save_images las imágenes
path_folder = os.path.join(base_path, f"Results/surface")
create_folder(path_folder)



# the user have to posicionate the camera to observe the fringe projection 

patterns_to_project = run_camera_and_fringes_ui(freq,resize_w,resize_h,resize_w,resize_h,displacement,0,(3, 2))

#  try for testing the camera conection
try:
    init_camera()
    cam = functions.cam                      # variable for contolling camera features
    vmb = functions.vmb                     # vimba system
except:
    print("camera is not connected.")
    close_camera()
    exit()

# main loop
while True:

    shifted_frames = phase_shifting_loop(cam, vmb, patterns_to_project, path_folder)

    # post-proccesing
    # obatain significant ...phase, phasor, compensed_phase, modulated intensity map... from the acquired frames
    complex_matrix = phasor(shifted_frames)
    amplitude_from_complex = amplitude_from_phasor(complex_matrix)
    phase = phase_calculation_from_array(shifted_frames)

    save_images = input("save_images (yes) (no):") 
    # save_images
    if save_images.lower() == "si" or save_images.lower() == "s" or save_images.lower() == "yes" or save_images.lower() == "y" or save_images == "Si" or save_images == "SI" or save_images.lower() == "YES":
            
            if os.path.exists(path_folder):
                cv2.imwrite(path_folder+"/modulated_intensity_map.png", np.uint8(255*amplitude_from_complex/np.max(amplitude_from_complex)))
                cv2.imwrite(path_folder+"/wrapped_phase.png", phase)
                # realización de la compensacion
                compensation = input("Realize the phase compensation: (yes) (no)")
                if compensation.lower() == "yes":
                    compensed_phase = phase_compensation(phase)
                    cv2.imwrite(path_folder+"/compensed_phase.png", np.uint8(255*compensed_phase/np.max(compensed_phase)))
            print("Images succesfully saved")

    continue_the_retrieval = input("Make the retrieval again? (yes) (no)") 
    
    # evaluate if the program continues or not
    if continue_the_retrieval.lower() != 'yes':
        close_camera()
        break
    