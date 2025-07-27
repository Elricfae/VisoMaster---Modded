import torch
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
import onnx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.helpers.downloader import download_file
from app.helpers.miscellaneous import is_file_exists
class FaceSwappers:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        if not self.models_processor.models[arcface_model]:
            self.models_processor.models[arcface_model] = self.models_processor.load_model(arcface_model)

        if arcface_model == 'CSCSArcFace':
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(arcface_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image
        
    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.models_processor.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 112, 112)

        if arcface_model == 'Inswapper128ArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
        elif arcface_model == 'SimSwapArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = torch.div(img.to(torch.float32), 255.0)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
        else:
            cropped_image = img.permute(1,2,0).clone() #112,112,3
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            # Normalize
            img = torch.div(img, 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = self.models_processor.models[arcface_model].get_inputs()[0].name

        outputs = self.models_processor.models[arcface_model].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.models_processor.models[arcface_model].io_binding()
        io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[arcface_model].run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image


    def calc_inswapper_latent(self, source_embedding):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_inswapper(self, image, embedding, output):
        if not self.models_processor.models['Inswapper128']:
            self.models_processor.models['Inswapper128'] = self.models_processor.load_model('Inswapper128')

        io_binding = self.models_processor.models['Inswapper128'].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Inswapper128'].run_with_iobinding(io_binding)


    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_iss_swapper(self, image, embedding, output, version="A"):
        ISS_MODEL_NAME = f'InStyleSwapper256 Version {version}'
        if not self.models_processor.models[ISS_MODEL_NAME]:
            self.models_processor.models[ISS_MODEL_NAME] = self.models_processor.load_model(ISS_MODEL_NAME)
        
        io_binding = self.models_processor.models[ISS_MODEL_NAME].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[ISS_MODEL_NAME].run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        #latent /= np.linalg.norm(latent)
        latent = latent/np.linalg.norm(latent,axis=1,keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.models_processor.models['SimSwap512']:
            self.models_processor.models['SimSwap512'] = self.models_processor.load_model('SimSwap512')

        io_binding = self.models_processor.models['SimSwap512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['SimSwap512'].run_with_iobinding(io_binding)

    def calc_swapper_latent_hyper(self, source_embedding, version="A"):
        latent = source_embedding.reshape((1, -1))
        latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
        return latent

    def run_swapper_hyperswap256(self, image, embedding, output, version="A"):
        MODEL = f'Hyperswap256 Version {version}'
        if not self.models_processor.models[MODEL]:
            self.models_processor.models[MODEL] = self.models_processor.load_model(MODEL)
        session = self.models_processor.models[MODEL]
        io_binding = session.io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1, 3, 256, 256), buffer_ptr=image.data_ptr())
        #io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=tuple(embedding.shape), buffer_ptr=embedding.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0,element_type=np.float32, shape=(1, 3, 256, 256), buffer_ptr=output.data_ptr())
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        session.run_with_iobinding(io_binding)
