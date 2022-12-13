import torch
from torchvision import transforms

class FaceComparisonEndpoint(object):
    def __init__(self):
        # Load the face recognition model
        self.model = torch.load("face_recognition_model.pt")

        # Set up image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, input_image, database_image):
        # Preprocess the input image
        input_image = self.preprocess(input_image)

        # Preprocess the database image
        database_image = self.preprocess(database_image)

        # Compare the input image with the database image using the face recognition model
        if self.model(input_image, database_image) > 0.5:
            return "Matching"
        else:
            return "Not matching"
#To use this endpoint, you would first initialize it like this:          
endpoint = FaceComparisonEndpoint()

#Then, you could use it to compare an input image from the camera with an image in the database like this:
result = endpoint(camera_image, database_image)
#This would return "Matching" if the face in the input image matches the face in the database image, or "Not matching" if it does not
