import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from vmbpy import *
import vmbpy
from screeninfo import get_monitors
import sys
import cv2
from queue import Queue, Empty
from vmbpy import *
import os
import time

cam = None                      # variable for contolling camera features
vmb = None                     # vimba system


def fringe_generator(freq_ini: int, screen_width: int, screen_height: int):
    """
    fringe generator, select the frequency, angle and returns 
    a list of arrays that corresponds to individual phase-shifts
    of the generated pattern. Additionally the sixth image is for uniform illumination
    """
    phase_shift = [0, np.pi/2, np.pi, 3*np.pi/2,2*np.pi, 1]     # phase values 
    angle_ini = 0      # initial angle
     
    #  Meshgrid related to the resolution of the second screen to project the structured-pattern 
    x = np.linspace(-5, 5, screen_width)
    y = np.linspace(-5, 5, screen_height)
    X, Y = np.meshgrid(x, y)

    # initial rotation of the fringes 
    initial_rotation = X * np.cos(angle_ini) + Y * np.sin(angle_ini)

    # figure features
    fig_patron, ax_patron = plt.subplots(figsize=(15,7))
    fig_patron.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax_patron.axis('off')
    manager = plt.get_current_fig_manager()
    ax_patron.set_position([0, 0, 1, 1])  # hacer que el eje ocupe toda la figura (ventana)

    # figure + sliders
    fig_slider, (ax_freq, ax_angle) = plt.subplots(2, 1, figsize=(3, 2))
    plt.subplots_adjust(left=0.2, bottom=0.3, top=0.95)

    # dislpay initial pattern
    im = ax_patron.imshow( np.sin(2 * np.pi * freq_ini * initial_rotation), cmap='gray', extent=[min(x), max(x), min(y), max(y)], aspect='auto',vmin=-0.8,vmax=0.8)
    
    # slider features
    slider_freq = Slider(ax_freq, 'Frequency', valmin = 0.0, valmax = 5, valinit = freq_ini, valstep = 0.05)
    slider_angle = Slider(ax_angle, 'Rotation', valmin = 0, valmax = 2*np.pi, valinit = angle_ini, valstep = np.pi / 4)

    def update_pattern(phase_value):
        freq_updated = slider_freq.val        
        angle_updated = slider_angle.val    

        # updated pattern when a phase-shift has been applied
        if phase_value != 1:
            coord_rotadas = X * np.cos(angle_updated) + Y * np.sin(angle_updated)
            patron_updated = np.sin(2 * np.pi * freq_updated * coord_rotadas + phase_value)
        
        # uniform illumination
        else:
            coord_rotadas = X * np.cos(angle_updated) + Y * np.sin(angle_updated)
            patron_updated = np.sin( 0 * coord_rotadas + np.pi/4)

        # update de plot
        try:
            im.set_data(patron_updated)
            fig_patron.canvas.draw()
            fig_patron.canvas.flush_events()
        except Exception as e:
            pass  

        return patron_updated
    
    # calls the update_pattern function when sliders changes
    slider_freq.on_changed(update_pattern)
    slider_angle.on_changed(update_pattern)

    plt.show()  
                

    # list to store de fringe patterns to project
    projected_pattern = []

    for shifting in phase_shift:
        
        shifted_fringe = update_pattern(shifting)

        #  to convert in 8-bits image
        if shifting != 1:
            shifted_fringe = np.uint8(((shifted_fringe-np.min(shifted_fringe))/np.max(shifted_fringe-np.min(shifted_fringe)))*255)
        
        # store array in list
        projected_pattern.append(shifted_fringe)

    return projected_pattern


def phase_compensation(wrappedphase):
    """
    wrapped phase must be a 8-bit numpy array
    """     
    phase_image = np.double(wrappedphase)
    phase_image = phase_image.squeeze()

    phase_image = 2*np.pi*wrappedphase/255 - np.pi # convertion to (-pi,pi] values


    print("shape",np.shape(phase_image))
    print("min max",np.min(phase_image), np.max(phase_image))

    # create a phasor to make the compensation
    complex_image = np.exp(2*np.pi*1j * phase_image)

    # create the meshgrid for the polynomial compensation
    x = np.linspace(-5, 5, phase_image.shape[1])
    y = np.linspace(-5, 5, phase_image.shape[0])
    X, Y = np.meshgrid(x, y)

    # slider values  for the polynomial coefficients 
    global slider_value1
    global slider_value2
    global slider_value3
    global slider_value4
    global slider_value5
    slider_value1=0
    slider_value2=0
    slider_value3=0
    slider_value4=0
    slider_value5=0

    # creation of the polynomio
    def polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5):                  # FASE polynomial EN X,Y, 
        return slider_value1 * X + slider_value2*Y + slider_value3*X**2 + slider_value4 * X**2 + slider_value5

    # plots
    fig, ax = plt.subplots(figsize=(13, 8))                          
    ax.axis('off')
    fig.subplots_adjust(left=0, right=0.5, top=0.4, bottom=0.3)
    ax_original = plt.subplot(1, 1, 1)
    img_original = ax_original.imshow(np.angle(complex_image), extent=[-5, 5, -5, 5], cmap='gray')           # mapa de color de campo complejo
    ax_original.set_title("Compensación en X,Y")
    plt.tight_layout(pad=4.0)

    # sliders features
    slider_ax1 = plt.axes([0.2, 0.25, 0.03, 0.65])
    slider1 = Slider(slider_ax1, 'x', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax2 = plt.axes([0.1, 0.25, 0.03, 0.65])
    slider2 = Slider(slider_ax2, 'y', -2,2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax3=plt.axes([0.15, 0.25, 0.03, 0.65])
    slider3=Slider(slider_ax3, 'x^2', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax4=plt.axes([0.05, 0.25, 0.03, 0.65])
    slider4=Slider(slider_ax4, 'y^2', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax5=plt.axes([0.08, 0.25, 0.03, 0.65])
    slider5=Slider(slider_ax5, 'offset', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
 
 
    #update function when a slider changes
    def update(val):
        global slider_value1
        global slider_value2
        global slider_value3
        global slider_value5
        slider_value1 = slider1.val
        slider_value2 = slider2.val
        slider_value3 = slider3.val
        slider_value4 = slider4.val
        slider_value5 = slider5.val

        # evaluation of the polynomial compensation 
        updated_complex_image = np.multiply(np.exp(1j * phase_image),np.exp(1j * 2 * np.pi * polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5) / 0.4))
        img_original.set_data(np.angle(updated_complex_image)) # update the plot with the compensed polynomial
        fig.canvas.draw_idle()

    # sliders
    slider1.on_changed(update)       
    slider2.on_changed(update)
    slider3.on_changed(update)       
    slider4.on_changed(update)
    slider5.on_changed(update)
    plt.show()

    print("coeff_values: ",slider_value1, slider_value2, slider_value3,slider_value4, slider_value5)

    # creation of the compensed phase with the adecuate polynomial coef values
    compensed = ((np.angle(np.multiply(np.exp(1j * phase_image),np.exp(1j * 2 * np.pi * polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5) / 0.4))))+np.pi)/(2*np.pi)*255


    return compensed


def desenvolvimiento(wrapped_phase):
    """ phase unwrraping
    input: a wrapped phase array-8bits
    """
    # convertion to 8-bits
    image = np.double(wrapped_phase)
    image = np.uint8(image)

    # convertion to values between (-pi,pi]
    image_wrapped = (image/255)*2*np.pi-np.pi   
    
    # to computate the unwrapping
    image_unwrapped = unwrap_phase(image_wrapped)

    # display the unwrapping array
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(14,9))
    ax1.imshow(wrapped_phase, cmap='gray')
    ax1.set_title('Original phase')
    ax2.imshow(image_unwrapped, cmap='gray')
    ax2.set_title('Unwrapped phase')
    plt.show()

    return image_unwrapped


def phasor(array_of_frames):
    """
    creation of phasor from array of frames (I0-I2)+j(I3-I1)
    or making the redundance operation with five arrays: I0+I4-2*I2  + 2*j(I3-I1)
    """
    #normalización del array
    normalization = [(im/(np.max(im))) for im in array_of_frames] 

    # Real part
    parte_real = (normalization[0] + normalization[4] - 2*normalization[2])

    # imag part
    parte_imaginaria = 2*1j*(normalization[3]-normalization[1])

    return parte_real + parte_imaginaria


def amplitude_from_phasor(complex_phasor):  
    """
    modulated_intensity map from complex phasor
    """
    # abs value of a+ib --> sqrt(b^2+a^2)
    amplitude=((np.abs(complex_phasor))/(np.max(np.abs(complex_phasor))))*255
    return amplitude


def amplitude_from_frames(array_of_frames):  
    """
    Modulated intensity map from frames
    """
    #normalization
    normalization = [(im/(np.max(im))) for im in array_of_frames]
    shifted13 = normalization[1]-normalization[3]
    shifted02 = normalization[0]-normalization[2]
    squared  = shifted13**2 + shifted02**2
    amplitude = np.sqrt(squared) 
    amplitude=((amplitude)/(np.max(amplitude)))*255
    return amplitude


def phase_calculation_from_complex_phasor(complex_phasor): 
    """
    for obtaining the phase is required an array of the complex phasor 
    """
    #Computation of wrapped phase with arctan2 that returns a matrix with values bewteen [-pi,pi)
    wrapped_phase = np.arctan2(np.imag(complex_phasor),np.real(complex_phasor))
    # convertion to 8-bits
    wrapped_phase_8_bits = (((wrapped_phase + np.pi)/(2*np.pi))*255)

    return wrapped_phase_8_bits


def phase_calculation_from_array(array_of_frames):
    """
    Phase obtantion from the frames. A list of frames is required
    """
    # normalization of the captures. Now, the values are between [0,1]
    normalization=[(im/(np.max(im))) for im in array_of_frames] 

    # Real array
    parte_real = (normalization[0]+normalization[4]-2*normalization[2])

    # Imag array
    parte_imaginaria = 2*(normalization[3]-normalization[1])

    # Convertion 8-bits
    parte_real = np.uint8( parte_real/np.max(parte_real)  )
    parte_imaginaria = np.uint8(( parte_imaginaria / np.max(parte_imaginaria) ))

    # phase [-pi,pi) values
    phase = np.arctan2((normalization[1]-normalization[3]),(normalization[0]-normalization[2]))

    # Convertion 8-bits
    phase = np.uint8(((phase+np.pi)/(2*np.pi))*255)

    return phase


def monitor_info():
    """
    displacement, resize_height, resize_width
    Get the main and second monitor info for resizing and moving the 
    fringe pattern to the second screen
    """

    monitors = get_monitors()
    number_of_monitors = int(np.shape(monitors)[0])

    for i in range(0,number_of_monitors):
        if monitors[i].is_primary:
            main_monitor = monitors[i]
        else:
            second_monitor = monitors[i]
    window_displacement = int(main_monitor.width)
    try:
        resize_window_height = int(second_monitor.height)
        resize_window_width = int(second_monitor.width)
        
       
    except NameError:
        print('No se detecto ningun monitor segundario')
        sys.exit(1)          # termina el programa con código de salida 1
    
    return window_displacement, resize_window_height, resize_window_width
        

def positionate_camera():
    """
    calibrate camera  position
    """
    WIN = 'camera'
    q = Queue(maxsize=1) 
    running = True

    def frame_handler(cam: Camera, stream: Stream, frame: Frame):
        """Callback de VmbPy: convertir a Mono8, pasar a OpenCV y reencolar el frame."""
        try:
           
            frame.convert_pixel_format(PixelFormat.Mono8)
            img = frame.as_opencv_image().copy()  
            
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(img)
        finally:
            stream.queue_frame(frame)  # to continue receiving frames

    with VmbSystem.get_instance() as vmb:           # context of vimba
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No hay cámaras detectadas.")
        with cams[0] as cam:                        # opens the camera in the vimba context
    
            cam.start_streaming(frame_handler)

            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL) # create de window once

            try:
                while True:
                    # close the window 
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    try:
                        img = q.get(timeout=0.01)  # wait for the frame
                        cv2.imshow(WIN, img)
                    except Empty:
                        # continues if the frames is not updated
                        pass

                    # ESC for continue and close 
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
            finally:
                cam.stop_streaming()
                cv2.destroyAllWindows()


def init_camera():
    '''
    Inicializa la cámara y establece los parámetros necesarios.
    '''
    global cam
    global vmb
    if vmb is None:
        vmb = vmbpy.VmbSystem.get_instance()  # Inicializar el sistema Vimba
        vmb.__enter__()
        cams = vmb.get_all_cameras()  # Obtener todas las cámaras disponibles
        cam = cams[0]  # Seleccionar la primera cámara disponible
        cam.__enter__()

        # configuración de las características de la cámara
    
        fr_enable = cam.get_feature_by_name('AcquisitionFrameRateEnable')
        fr_enable.set(True)
       

        cam.get_feature_by_name('ExposureAuto').set('Off')
        cam.get_feature_by_name('ExposureTime').set(60000.0)  # microsegundos
        cam.get_feature_by_name('GainAuto').set('Off')
        cam.get_feature_by_name('Gain').set(0.0)  
        cam.get_feature_by_name('BalanceWhiteAuto').set('Off')
        cam.set_pixel_format(PixelFormat.Mono8) 
        frame_rate = cam.get_feature_by_name('AcquisitionFrameRate')
        max_posible_frame_rate = frame_rate.get_range()[1]
        try:
            frame_rate.set(max_posible_frame_rate)
        except:
            pass


def close_camera():
    '''
    Cierra la cámara correctamente.
    '''
    global cam
    global vmb
    cam.__exit__(None, None, None)
    vmb.__exit__(None, None, None)


def create_folder(path_folder: str):


    """string with the path of the newfolder"""
    # Create the folder if doesn't exist 
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)


def display_pattern_second_screen(pattern, resize_w, resize_h, displacement):
    """
    display de sinusoidal pattern on second screen using the pattern input
    """
    # display the sinusoidal pattern on the second screen
    cv2.imshow('pattern',pattern)
    cv2.resizeWindow('pattern', resize_w, resize_h)
    cv2.moveWindow('pattern', displacement,0)
    cv2.waitKey(1)



def phase_shifting_loop(cam, vmb, patterns_to_project, path_folder):
    
    """
    Loop where the acquisition of shifted-pattern frames will happen
    """
    print("Frames acquisition has started...")

    delay_camera_screen  = 500E-3   # delay to sinchronize screen projection and camera acquisition
    shifted_frames = []             # List for storing the acquired frames
    fringe_shift_counter = 0        # fringe shift counter to guarantee the phase shifting-technique
    while True:
        cv2.imshow('pattern',((patterns_to_project[fringe_shift_counter])))
        cv2.waitKey(1)
        time.sleep(delay_camera_screen)

        # to save image under uniform illumination
        if fringe_shift_counter == 5:
            cam.set_pixel_format(PixelFormat.Rgb8) 
            frame = cam.get_frame()
            frame = frame.as_numpy_ndarray()
            frame = np.squeeze(frame)
            cv2.imwrite(path_folder+f"/Phase_{fringe_shift_counter}color.png", frame)

        # acquiring and saving the shifted-phase
        cam.set_pixel_format(PixelFormat.Mono8) 
        frame = cam.get_frame()
        frame = frame.as_numpy_ndarray()
        frame = np.squeeze(frame)
        cv2.imwrite(path_folder + f"/Phase_{fringe_shift_counter}.png", frame)

        # storing the frame for post-processing
        shifted_frames.append(frame)
        fringe_shift_counter +=1

        # finish at the sixth capture
        if fringe_shift_counter == 6 :
            cv2.destroyAllWindows()
            break
    return shifted_frames



def run_camera_and_fringes_ui(
    freq_ini: float,
    screen_width: int,
    screen_height: int,
    pattern_resize_w: int,
    pattern_resize_h: int,
    pattern_displacement_x: int = 1920,   # move 1920 pixels the second screen
    pattern_displacement_y: int = 0,
    slider_fig_size=(3, 2)
):
    """
    -Slider(freq, angle of fringes in main window)
    -Projected fringes on second screen
    -OpenCV visualization of the camera 
    -Close any window o press ESC and the code will continue
    -returns the list of projected pattern to display and make the phase shift
    -5 sinusoidal patterns with phase: [0, π/2, π, 3π/2, 2π]  and 1 for uniform illumination
    """

    # geometrý 
    # Meshgrid
    x = np.linspace(-5, 5, screen_width)
    y = np.linspace(-5, 5, screen_height)
    X, Y = np.meshgrid(x, y)

    # sliders state
    state = {
        'freq': float(freq_ini),
        'angle': 0.0,
        'stop': False,
    }

    #  slider fig (Matplotlib)
    fig_sliders, (ax_freq, ax_angle) = plt.subplots(2, 1, figsize=slider_fig_size)
    plt.subplots_adjust(left=0.2, bottom=0.3, top=0.95)
    fig_sliders.canvas.manager.set_window_title('Fringe Controls')

    slider_freq = Slider(ax_freq, 'Frequency', valmin=0.0, valmax=5.0,
                         valinit=freq_ini, valstep=0.05)
    slider_angle = Slider(ax_angle, 'Rotation', valmin=0.0, valmax=2*np.pi,
                          valinit=0.0, valstep=np.pi/4)

    def on_change(_):
        state['freq'] = float(slider_freq.val)
        state['angle'] = float(slider_angle.val)
    slider_freq.on_changed(on_change)
    slider_angle.on_changed(on_change)

    def on_close_matplotlib(event):
        state['stop'] = True
    fig_sliders.canvas.mpl_connect('close_event', on_close_matplotlib)

    # display the slider plot and do not block the code
    plt.show(block=False)

    #  WINDOWS OpenCV (pattern and camera) 
    # Ventana de franjas en segunda pantalla
    WIN_PATTERN = 'pattern'
    cv2.namedWindow(WIN_PATTERN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PATTERN, pattern_resize_w, pattern_resize_h)
    cv2.moveWindow(WIN_PATTERN, pattern_displacement_x, pattern_displacement_y)

    # camera window
    WIN_CAMERA = 'camera'
    cv2.namedWindow(WIN_CAMERA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WIN_CAMERA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # asynchronous for vimba
    q = Queue(maxsize=1)

    def frame_handler(cam: Camera, stream: Stream, frame: Frame):
        """Callback corto: convierte a Mono8, copia numpy y reencola; siempre requeue."""
        try:
            frame.convert_pixel_format(PixelFormat.Mono8)
            img = frame.as_opencv_image().copy()
            # Dejar solo el más reciente
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(img)
        finally:
            stream.queue_frame(frame)

    # context
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            plt.close(fig_sliders)
            cv2.destroyAllWindows()
            raise RuntimeError("no detected camera.")

        with cams[0] as cam:
            cam.start_streaming(frame_handler)

            try:
                while not state['stop']:
                    # close if any window is closed
                    if cv2.getWindowProperty(WIN_CAMERA, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.getWindowProperty(WIN_PATTERN, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    # ==== Camera ====
                    try:
                        img = q.get(timeout=0.01)
                        cv2.imshow(WIN_CAMERA, img)
                    except Empty:
                        pass

                    # ==== fringes
                    coord_rot = X * np.cos(state['angle']) + Y * np.sin(state['angle'])
                    patt = np.sin(2 * np.pi * state['freq'] * coord_rot)  # fase 0
                    # Normalizar a 8 bits para visualización con OpenCV
                    patt_u8 = ((patt - patt.min()) / (patt.max() - patt.min() + 1e-12) * 255.0).astype(np.uint8)
                    cv2.imshow(WIN_PATTERN, patt_u8)

                    # ==== events 
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break

                    plt.pause(0.001)

            finally:
                cam.stop_streaming()

    # close the cv2 windows and matplotlib ones
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        plt.close(fig_sliders)
    except Exception:
        pass

    
    # generation of patterns
    freq_final = state['freq']
    angle_final = state['angle']
    phase_shift = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 1]  # el "1" indicates uniform illumination

    projected_pattern = []
    coord_rot_final = X * np.cos(angle_final) + Y * np.sin(angle_final)

    for shifting in phase_shift:
        if shifting != 1:
            pat = np.sin(2 * np.pi * freq_final * coord_rot_final + shifting)
            pat_u8 = np.uint8(((pat - pat.min()) / (pat.max() - pat.min() + 1e-12)) * 255.0)
            projected_pattern.append(pat_u8)
        else:
            # uniform illumination
            pat_uniform = np.sin(0 * coord_rot_final + np.pi/4)
            projected_pattern.append(pat_uniform)  
    return projected_pattern
