import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)

model1 = timm.create_model('tf_efficientnet_b8', pretrained=True)
model2 = timm.create_model('resnet50', pretrained=True)
model3 = timm.create_model('tv_resnet34', pretrained=True)