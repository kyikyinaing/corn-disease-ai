from ml_model import CornDiseaseModel

m = CornDiseaseModel()
print(m.predict_from_image("sample.jpg"))