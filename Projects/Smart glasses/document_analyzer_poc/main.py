import cv2

class DocumentAnalyzer:
    """Pipeline for analyzing a document from image to speech."""
    
    def __init__(self):
        """Initialization the Document Analyzer instance"""
        pass


    def _capture_image(self):
        """Simulates capturing an image from a camera"""
        
        print("Simulating image capture from camera...")
        video_capture = cv2.VideoCapture(0) # Conneect to the default camera - index 0

        while True:
            ret, frame = video_capture.read() # -> return_value, frame
            cv2.imshow('New frame', frame)
            key = cv2.waitKey(1) # Wait for one miliseecond for a pressed key
            cv2.imwrite("from_camera.png", frame)
            video_capture.release() # Closes video file or capturing device 
            cv2.destroyAllWindows()
            break
        dummy_path = "Repozytorium VSC\Projects\Smart glasses\document_analyzer_poc\dummy_text.png"
        return dummy_path


    def _extract_text_from_image(self, image_path: str) -> str:
        """Extracting text from a given image path"""
        pass


    def _summarize_text(self, text: str)-> str:
        """Summarize of the provided text."""
        pass


    def _speak_summary(self, summary):
        """Speaking the summary out loud"""
        pass


    def analyze_document(self):
        """Runs the full document analysis pipeline"""
        image_path = self._capture_image()
        extracted_text = self._extract_text_from_image(image_path)
        summary = self._summarize_text(extracted_text)
        self._speak_summary(summary)


if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    analyzer.analyze_document()
