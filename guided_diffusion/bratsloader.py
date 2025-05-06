import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
      '''
      directory is expected to contain some folder structure like:
      ├── Patient1
      │   ├── Patient1_t1c.nii.gz
      │   ├── Patient1_t1n.nii.gz
      │   ├── Patient1_t2f.nii.gz
      │   ├── Patient1_t2w.nii.gz
      │   ├── Patient1_seg.nii.gz
      '''
      super().__init__()
      self.directory = os.path.expanduser(directory)
      self.transform = transform
      self.test_flag = test_flag

      if test_flag:
          self.seqtypes = ['t1c', 't1n', 't2f', 't2w']
      else:
          self.seqtypes = ['t1c', 't1n', 't2f', 't2w', 'seg']

      self.seqtypes_set = set(self.seqtypes)
      self.database = []
      for root, dirs, files in os.walk(self.directory):
          # if there are files in the folder, process them
          if files:
              files.sort()
              datapoint = dict()
              for f in files:
                  # Extract the sequence type from the filename
                  seqtype = f.split('-')[-1].split('.')[0]
                  if seqtype in self.seqtypes_set:
                      datapoint[seqtype] = os.path.join(root, f)
              assert set(datapoint.keys()) == self.seqtypes_set, \
                  f'Datapoint is incomplete in {root}, found: {datapoint.keys()}'
              self.database.append(datapoint)


    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, test_flag=False):
        '''
        directory is expected to contain some folder structure like:
        ├── Patient1
        │   ├── Patient1_t1c.nii.gz
        │   ├── Patient1_t1n.nii.gz
        │   ├── Patient1_t2f.nii.gz
        │   ├── Patient1_t2w.nii.gz
        │   ├── Patient1_seg.nii.gz
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag

        if test_flag:
            self.seqtypes = ['t1c', 't1n', 't2f', 't2w']
        else:
            self.seqtypes = ['t1c', 't1n', 't2f', 't2w', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # if there are files in the folder, process them
            if files:
                files.sort()
                datapoint = dict()
                # print(f"[{root}] Found files: {files}")
                for f in files:
                    if f.endswith('.nii.gz'):
                        # Use more robust parsing
                        for s in self.seqtypes:
                            if f.endswith(f"{s}.nii.gz"):
                                # print(f"Processing file: {f}, extracted seqtype: {s}")
                                datapoint[s] = os.path.join(root, f)

                # for f in files:
                #     # Extract the sequence type from the filename
                #     seqtype = f.split('-')[-1].split('.')[0]
                #     print(f"Processing file: {f}, extracted seqtype: {seqtype}")

                #     if seqtype in self.seqtypes_set:
                #         datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f"Datapoint is incomplete in {root}, found: {datapoint.keys()}"
                self.database.append(datapoint)

    def __len__(self):
        # Each patient volume typically contains 155 slices
        return len(self.database) * 155

    def __getitem__(self, index):
        out = []
        # Determine which patient volume and slice to load
        patient_index = index // 155
        slice_index = index % 155
        filedict = self.database[patient_index]

        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path = filedict[seqtype]
            # Extract the slice along the Z-axis
            # print(f"Loading slice {slice_index} from volume shape {nib_img.get_fdata().shape}")
            slice_data = torch.tensor(nib_img.get_fdata())[:, :, slice_index]
            if seqtype != 'seg':
                slice_data = (slice_data - slice_data.mean()) / (slice_data.std() + 1e-8) # Intensity Normalization
            out.append(slice_data)

        out = torch.stack(out)

        if self.test_flag:
            image = out
            # Optional cropping (uncomment if needed)
            # image = image[..., 8:-8, 8:-8]
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + f"_slice{slice_index}.nii")  # virtual path
        else:
            image = out[:-1, ...]  # All modalities except 'seg'
            label = out[-1, ...][None, ...]  # The 'seg' modality
            # Optional cropping (uncomment if needed)
            # image = image[..., 8:-8, 8:-8]
            # label = label[..., 8:-8, 8:-8]
            # 🔍 DEBUG: print intensity stats for each modality
            # for i, modality in enumerate(image):
                # print(f"Modality {self.seqtypes[i]}: min={modality.min().item():.2f}, max={modality.max().item():.2f}, mean={modality.mean().item():.2f}, std={modality.std().item():.2f}")

            # print("label values:", torch.unique(label))  # Check current label values tobe deleted
            label[label == 3] = 4
            label = torch.where(label > 0, 1, 0).float()  # Merge all tumor classes into one
            print("label values after merge:", torch.unique(label))
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            # print("image shape:", image.shape)
            # print("label shape:", label.shape)
            return (image, label, path.split('.nii')[0] + f"_slice{slice_index}.nii")  # virtual path






