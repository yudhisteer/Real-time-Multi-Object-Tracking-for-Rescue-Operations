import cv2
import os

def background_subtraction(file: str, save: bool=False, output_video_path: str='output_video.mp4') -> None:

    cap = cv2.VideoCapture(file)

    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=350, detectShadows=True)

    # Get the video frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Threshold to remove shadows
        _, fg_mask = cv2.threshold(fg_mask, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

        # Display the original and subtracted frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        # Write the subtracted frame to the output video
        if save:
            out.write(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))

        keyboard = cv2.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # Go back to the parent directory
    parent_directory = os.path.dirname(os.getcwd())

    # Set input directory
    video_file = os.path.join(parent_directory, 'Data', 'Occlusion.mp4')

    # Background subtraction:
    background_subtraction(video_file, save=True, output_video_path="Occlusion _bg.mp4")