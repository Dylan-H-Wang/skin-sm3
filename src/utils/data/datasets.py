import os
import logging

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image

from .functional import opencv_image_loader as load_img
from .functional import get_color_distortion


# Modify from https://github.com/jeremykawahara/derm7pt
class SevenPCDataset():
    # 'names': the name of the tag associated with the image.
    # 'abbrevs': a unique abbreviation. Used as the key to link other DataFrames.
    # 'colnames': the name of the column in the CSV that corresponds to this tag.
    # 'seven_pt': 1 if part of the 7-point criteria. Else 0 (diagnosis get's 0).

    # These are the `categories` (i.e., diagnosis + 7pt checklist) from the JBHI paper.
    tags = pd.DataFrame([
        {'names': 'Diagnosis', 'abbrevs': 'DIAG', 'colnames': 'diagnosis', 'seven_pt': 0},
        {'names': 'Pigment Network', 'abbrevs': 'PN', 'colnames': 'pigment_network', 'seven_pt': 1},
        {'names': 'Blue Whitish Veil', 'abbrevs': 'BWV', 'colnames': 'blue_whitish_veil', 'seven_pt': 1},
        {'names': 'Vascular Structures', 'abbrevs': 'VS', 'colnames': 'vascular_structures', 'seven_pt': 1},
        {'names': 'Pigmentation', 'abbrevs': 'PIG', 'colnames': 'pigmentation', 'seven_pt': 1},
        {'names': 'Streaks', 'abbrevs': 'STR', 'colnames': 'streaks', 'seven_pt': 1},
        {'names': 'Dots and Globules', 'abbrevs': 'DaG', 'colnames': 'dots_and_globules', 'seven_pt': 1},
        {'names': 'Regression Structures', 'abbrevs': 'RS', 'colnames': 'regression_structures', 'seven_pt': 1},
    ])

    # Each `category` has several `labels` associated with it. Each `label` has several properties.
    # `nums`: an integer unique within a category, used for the neural network.
    # `names`: A string that corresponds to the values within the csv file that represent the type of label.
    #       Multiple labels can be grouped to form a single label by passing in a list of strings.
    # `abbrevs`: a unique abbreviation that represents the label.
    # `info`: include other helpful info for yourself here. Not otherwise used.
    #         I kept forgetting what some diseases were, so included additional info for a few.
    # Note there is no `scores` since the diagnosis is an overall label of the image,
    # and does not contribute to the 7-point checklist score.
    diagnosis = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1, 'names': 'blue nevus', 'abbrevs': 'BLN'},
        {'nums': 2, 'names': 'clark nevus', 'abbrevs': 'CN'},
        {'nums': 3, 'names': 'combined nevus', 'abbrevs': 'CBN'},
        {'nums': 4, 'names': 'congenital nevus', 'abbrevs': 'CGN'},
        {'nums': 5, 'names': 'dermal nevus', 'abbrevs': 'DN'},
        {'nums': 6, 'names': 'dermatofibroma', 'abbrevs': 'DF'},
        {'nums': 7, 'names': 'lentigo', 'abbrevs': 'LT'},
        {'nums': 8, 'names': ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                              'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)',
                              'melanoma metastasis'], 'abbrevs': 'MEL'},
        {'nums': 9, 'names': 'melanosis', 'abbrevs': 'MLS', 'info': 'Hyperpigmentation of the skin.'},
        {'nums': 10, 'names': 'miscellaneous', 'abbrevs': 'MISC'},
        {'nums': 11, 'names': 'recurrent nevus', 'abbrevs': 'RN'},
        {'nums': 12, 'names': 'reed or spitz nevus', 'abbrevs': 'RSN'},
        {'nums': 13, 'names': 'seborrheic keratosis', 'abbrevs': 'SK'},
        {'nums': 14, 'names': 'vascular lesion', 'abbrevs': 'VL'},
    ])

    # `scores`: An integer that represents how much the label contributes to
    #           the 7-point checklist score.
    pigment_network = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'typical', 'abbrevs': 'TYP', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'atypical', 'abbrevs': 'ATP', 'scores': 2, 'info': ''},
    ])

    blue_whitish_veil = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'present', 'abbrevs': 'PRS', 'scores': 2, 'info': ''},
    ])

    vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'arborizing', 'abbrevs': 'ARB', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'comma', 'abbrevs': 'COM', 'scores': 0, 'info': ''},
        {'nums': 3, 'names': 'hairpin', 'abbrevs': 'HP', 'scores': 0, 'info': ''},
        {'nums': 4, 'names': 'within regression', 'abbrevs': 'WR', 'scores': 0, 'info': ''},
        {'nums': 5, 'names': 'wreath', 'abbrevs': 'WTH', 'scores': 0, 'info': ''},
        {'nums': 6, 'names': 'dotted', 'abbrevs': 'DOT', 'scores': 2, 'info': ''},
        {'nums': 7, 'names': 'linear irregular', 'abbrevs': 'LIR', 'scores': 2, 'info': ''},
    ])

    pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'diffuse regular', 'abbrevs': 'DR', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'localized regular', 'abbrevs': 'LR', 'scores': 0, 'info': ''},
        {'nums': 3, 'names': 'diffuse irregular', 'abbrevs': 'DI', 'scores': 1, 'info': ''},
        {'nums': 4, 'names': 'localized irregular', 'abbrevs': 'LI', 'scores': 1, 'info': ''},
    ])

    streaks = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
    ])

    dots_and_globules = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
    ])

    regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'blue areas', 'abbrevs': 'BA', 'scores': 1, 'info': ''},
        {'nums': 2, 'names': 'white areas', 'abbrevs': 'WA', 'scores': 1, 'info': ''},
        {'nums': 3, 'names': 'combinations', 'abbrevs': 'CMB', 'scores': 1, 'info': ''},
    ])

    def __init__(self, dir_images, metadata_df, train_indexes, valid_indexes, test_indexes, crop_amount=25):
        """The meta-data for the Derm7Pt dataset.
        Args:
            dir_images: A string indicating the root directory of the images.
            metadata_df: A Pandas data-frame that contains all the meta-data for each case.
            train_indexes: A list of integers that represent training indexes into metadata_df.
            valid_indexes: 
            test_indexes:
            crop_amount: An integer specifying how many pixels to crop at the image border.
                Useful if images contain a black boundary.
        """

        self.derm_column = 'derm'
        self.clinic_column = 'clinic'

        # Copy data-frame as is modified by-reference.
        self.df = metadata_df.copy()
        self.dir_imgs = dir_images
        self.crop_amount = crop_amount

        # Modify meta to include numeric labels in the columns.
        self.set_df_numeric_labels()

        # Check the properties of the class match the columns in self.df
        self.check_myself()

        # Make sure all the indexes are in at least one fold.
        match_indexes = np.alltrue(np.sort(np.concatenate((train_indexes, valid_indexes, test_indexes)))
                                   == range(len(self.df)))
        if not match_indexes:
            print("Warning! The train/valid/test indexes do not match the total number of samples.")

        all_indexes = np.concatenate((train_indexes, valid_indexes, test_indexes))
        assert len(set(all_indexes)) == len(all_indexes), "Error! There are duplicate indexes in train, valid, or test."

        self.train = self.df.iloc[train_indexes]
        self.valid = self.df.iloc[valid_indexes]
        self.test = self.df.iloc[test_indexes]

        # Meta data.
        self.elevation_dict = self.get_dict_labels(self.df.elevation)
        self.sex_dict = self.get_dict_labels(self.df.sex)
        self.location_dict = self.get_dict_labels(self.df.location)

    def set_df_numeric_labels(self):
        """Add numeric values to the columns in the df."""
        for abbrev in self.tags.abbrevs:
            col_name = self.get_column_name(abbrev)
            nums = self.strings2numeric(self.df[col_name], self.get_label_names(abbrev), self.get_label_nums(abbrev))
            col_name_numeric = self.get_column_name_numeric(abbrev)
            self.df[col_name_numeric] = nums

    def get_column_name(self, abbrev):
        """Return the data-frame column name that corresponds to the string labels for this `abbrev`."""
        tag = self.get_tag_by_abbrev(abbrev)
        return tag.colnames.iloc[0]

    def get_tag_by_abbrev(self, abbrev):
        """Return the label info for a given abbreviation."""
        tag = self.tags[self.tags.abbrevs == abbrev]
        if tag.empty:
            raise ValueError('Error: no tag for `%s`' % str(abbrev))

        return tag

    def get_tag_name(self, abbrev):
        tag = self.get_tag_by_abbrev(abbrev)
        return tag.names.iloc[0]

    def get_label_names(self, abbrev, ignore_sub_names=False):
        """Return the names for all the labels for a given category abbrev."""
        lab = self.get_label_by_abbrev(abbrev)
        label_names = []
        for name in lab.names.values:
            if ignore_sub_names:
                if type(name) is list:
                    # Use the first name in the list if is a list.
                    name = name[0]

            label_names.append(name)

        return label_names

    def get_label_by_abbrev(self, abbrev):
        tag = self.get_tag_by_abbrev(abbrev)
        lab = getattr(self, tag.colnames.values[0])
        return lab

    def get_label_abbrevs(self, abbrev):
        lab = self.get_label_by_abbrev(abbrev)
        return lab.abbrevs

    def get_label_nums(self, abbrev):
        lab = self.get_label_by_abbrev(abbrev)
        return lab.nums.values

    def get_column_name_numeric(self, abbrev):
        """Return the name of the data-frame column that corresponds to the numeric labels for this `abbrev`."""
        col_name = self.get_column_name(abbrev)
        return col_name + '_numeric'

    @staticmethod
    def get_dict_labels(df_column_names):

        label_names = df_column_names.unique()
        label_names.sort()
        label_dict = {}
        for label_idx, label_name in enumerate(label_names):
            label_dict[label_name] = label_idx

        return label_dict

    def check_myself(self):
        """Check the properties of the class match the columns in self.df"""
        for var_name in self.tags.colnames:

            # Check the column names correspond to the attributes of the class.
            var = getattr(self, var_name, None)
            if var is None:
                raise ValueError('Error: missing attribute `self.%s`. Make sure you manually set it.' % str(var_name))

            # Check that all the column names correspond to a column name in the CSV.
            if var_name not in self.df.columns:
                raise ValueError('Error: the variable name `%s` does not link to a column name.' % str(var_name))

    def n_samples(self):
        """Number of samples/lesions in the dataset."""
        return len(self.df)

    def dataset_stats(self):

        n_train = len(self.train)
        n_test = len(self.test)
        n_valid = len(self.valid)

        # stats = {'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test}

        print('Number of cases: ' + str(self.n_samples()))
        print('Number of cases to train: ' + str(n_train))
        print('Number of cases to validate: ' + str(n_valid))
        print('Number of cases to test: ' + str(n_test))

        assert n_train + n_test + n_valid == self.n_samples(), \
            "The train+test+valid cases do not equal the total cases!"

    def get_data_type(self, data_type):
        if data_type == 'all':
            df = self.df
        elif data_type == 'train':
            df = self.train
        elif data_type == 'valid':
            df = self.valid
        elif data_type == 'test':
            df = self.test
        else:
            raise ValueError('Error: data_type: `' + str(data_type) + '` unknown option.')

        return df

    def get_tag_abbrevs(self):
        """Return the abbreviations for all the tags."""
        return self.tags.abbrevs

    def labels2hot(self, labels, abbrev='DIAG'):
        """
        Convert the labels to 1-hot encoding.
        Args:
            labels: a list or array of numeric labels, e.g., labels=[1,0,1].
            abbrev:
        Returns: The labels one-hot-encoded.
        """
        nb_classes = len(self.get_label_nums(abbrev))
        one_hot_labs = np.eye(nb_classes)[labels]
        # one_hot_labs = keras.utils.np_utils.to_categorical(labels, nb_classes)
        return one_hot_labs

    def get_labels(self, data_type='all', one_hot=False):
        """
        Return all the numeric class labels.
        Args:
            data_type:
            one_hot: Boolean, where if `True`, then encode the labels as 1-hot encoding.
        Returns: a dictionary of all the class tags.
        """

        df = self._get_data_frame(data_type)
        Y = {}

        for abbrev in self.get_tag_abbrevs():
            labels = df[self.get_column_name_numeric(abbrev)]
            if one_hot:
                Y[abbrev] = self.labels2hot(labels=labels, abbrev=abbrev)
            else:
                Y[abbrev] = labels

        return Y

    def get_label_names_abbrev(self, abbrev):
        abbrevs = self.get_label_abbrevs(abbrev)
        full_names = self.get_label_names(abbrev, ignore_sub_names=True)

        names_abbrev = []
        for abbrev, name in zip(abbrevs, full_names):
            names_abbrev.append(name + ' (' + abbrev + ')')

        return names_abbrev

    def _get_data_frame(self, data_type='all'):
        if data_type == 'all':
            df = self.df
        elif data_type == 'train':
            df = self.train
        elif data_type == 'valid':
            df = self.valid
        elif data_type == 'test':
            df = self.test
        else:
            raise ValueError('Error: data_type: `' + str(data_type) + '` unknown option.')

        return df

    def get_img_paths(self, data_type='all', img_type='derm'):
        """
        Return the paths to the images.
        Args:
            data_type: must be one of: 'all', 'train', 'valid', or 'test'
            img_type: must be one of: 'derm', or 'clinic'
        Returns:
            A list of the full paths to each image.
        """
        df = self._get_data_frame(data_type)

        if img_type == 'derm':
            img_names = df.derm
        elif img_type == 'clinic':
            img_names = df.clinic
        else:
            raise ValueError('Error: img_type `' + str(img_type) + '` is an unknown option.')

        return [os.path.join(self.dir_imgs, img_path) for img_path in img_names]

    def _get_image(self, row_index, image_type, crop_amount=None):

        if image_type == 'derm':
            img_name = self.derm_img_name(row_index)
        elif image_type == 'clinic':
            img_name = self.clinic_img_name(row_index)
        else:
            raise ValueError("Unknown `image_type`.")

        if crop_amount is None:
            crop_amount = self.crop_amount

        # Must be >= 0.
        assert crop_amount >= 0

        img = np.asarray(load_img(img_name))

        # Make sure there are only 3 dimensions in the color channel.
        img = img[:, :, :3]

        # Some images have a surrounding black border, so remove `crop_amount` pixels around the entire image.
        if crop_amount > 0:
            img = img[crop_amount:-crop_amount, crop_amount:-crop_amount, :]

        return img

    def derm_img_name(self, row_index):
        """Returns the path and name of the image in the `idx` row of the meta-data.
        Args:
            row_index: An integer that specifies the index of the row within the meta-data.
        Returns:
            A string that represents the path and filename to the image.
        """
        return os.path.join(self.dir_imgs, str(self.df.iloc[row_index][self.derm_column]))

    def clinic_img_name(self, idx):
        return os.path.join(self.dir_imgs, str(self.df.iloc[idx][self.clinic_column]))

    def derm_image(self, row_index, crop_amount=None, target_size=None):
        """Return the dermoscopic image that corresponds to the given row."""
        return self._get_image(row_index, 'derm', crop_amount, target_size=target_size)

    def clinic_image(self, row_index, crop_amount=None, target_size=None):
        """Return the clinical image that corresponds to the given row."""
        return self._get_image(row_index, 'clinic', crop_amount, target_size=target_size)

    def strings2numeric(self, strings, names, numeric_vals, sentinel=-1):
        """Convert strings to numeric values.
        Args:
            strings: a list of strings to convert to numeric tags.
            names: a list of unique strings that correspond to the numeric values.
            numeric_vals: a list of integers that correspond to the ordering in names.
            sentinel: a value that is not in the numeric_vals.
        Returns:
            a numpy array of the numeric values to use (instead of the strings)
        """

        if sentinel in numeric_vals:
            raise ValueError("`sentinel` should not occur in `numeric_val`.")

        strings = np.asarray(strings)

        numeric = np.ones(shape=len(strings), dtype=int) * sentinel

        for class_idx, label in zip(numeric_vals, names):
            # If this is a list, then group all these (sublist) items with the same numeric label.
            if isinstance(label, list):
                for l in label:
                    numeric[np.asarray(strings == l)] = class_idx

            else:  # This is a single sub-label already.
                numeric[np.asarray(strings == label)] = class_idx

        if np.any(numeric == sentinel):
            missing_indexes = np.where(numeric == sentinel)
            first_missing_str = strings[missing_indexes[0]]
            raise ValueError(
                "The value `%s` in `strings` do not exist in `names`. Did you spell something wrong?" % first_missing_str)

        return numeric
    

class SevenPCGroupDataset(SevenPCDataset):
    diagnosis = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV'},
        {'nums': 2,
         'names': ['melanoma', 'melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)',
                   'melanoma (more than 1.5 mm)', 'melanoma metastasis'], 'abbrevs': 'MEL'},
        {'nums': 3, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis',
                              'miscellaneous', 'vascular lesion'], 'abbrevs': 'MISC'},
        {'nums': 4, 'names': 'seborrheic keratosis', 'abbrevs': 'SK'},
    ])

    vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'],
         'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'abbrevs': 'IR', 'scores': 2,
         'info': ''},
    ])

    pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'diffuse regular', 'localized regular'], 'abbrevs': 'REG', 'scores': 0,
         'info': ''},
        {'nums': 2, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'abbrevs': 'IR', 'scores': 1,
         'info': ''},
    ])

    regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'], 'abbrevs': 'PRS', 'scores': 1,
         'info': ''},
    ])
    

class SevenPCBaseDataset(Dataset):
    LABEL_ORD = ["DIAG", "PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]

    def __init__(
        self,
        args,
        data_trans,
        mode,
        return_index=False,
        return_img_path=False
    ):
        super().__init__()
        logger = logging.getLogger(f"{args.logger_name}.{self.__class__.__name__}")

        self.data_path = args.data_path + "/images"
        self.meta_dir = args.data_path
        self.data_trans = data_trans
        self.mode = mode
        self.return_index = return_index
        self.return_img_path = return_img_path
       
        # Apply cropping around the boundary of the images to get rid of the black boundary.
        self.crop_amount = 25
        self._prepare()

        logger.info(
            f"============ Building {self.__class__.__name__} ============\n"
            f"- dataset split: {mode}\n"
            f"- length: {self.__len__()}"
        )

    def __getitem__(self, index):
        derm_image = self.derm_data[index]
        clinic_image = self.clinic_data[index]
        label = torch.as_tensor([self.labels[i].iloc[index] for i in self.__class__.LABEL_ORD])
        img_path = (derm_image, clinic_image)

        # derm_image = np.asarray(pil_image_loader(derm_image))[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        # clinic_image = np.asarray(load_img(clinic_image))[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        derm_image = Image.fromarray(load_img(derm_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount])
        clinic_image = Image.fromarray(load_img(clinic_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount])

        if self.data_trans is None:
            derm_trans = derm_image
            clinic_trans = clinic_image
        elif isinstance(self.data_trans, list):
            derm_trans = list(map(lambda trans: trans(derm_image), self.data_trans))
            clinic_trans = list(map(lambda trans: trans(clinic_image), self.data_trans))
        else:
            derm_trans = self.data_trans(derm_image)
            clinic_trans = self.data_trans(clinic_image)

        if self.return_index:
            return index, (derm_trans, clinic_trans, label)
        if self.return_img_path:
            return img_path, (derm_trans, clinic_trans, label)
        return (derm_trans, clinic_trans, label)

    def __len__(self):
        return len(self.derm_data)

    def _prepare(self):
        # elevation = df["elevation"]
        # location = df["location"]
        # sex = df["sex"]

        meta_df = pd.read_csv(os.path.join(self.meta_dir, "meta.csv"))
        train_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "train_indexes.csv"))["indexes"])
        valid_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "valid_indexes.csv"))["indexes"])
        test_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "test_indexes.csv"))["indexes"])

        derm_data = SevenPCGroupDataset(dir_images=self.data_path, 
                    metadata_df=meta_df.copy(), # Copy as is modified.
                    train_indexes=train_indexes, valid_indexes=valid_indexes, 
                    test_indexes=test_indexes)

        # label is 1 diagnosis + 7 critiera: ["PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]
        if self.mode == "train":
            self.derm_data = derm_data.get_img_paths(data_type='train', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='train', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='train', one_hot=False)
        elif self.mode == "val":
            self.derm_data = derm_data.get_img_paths(data_type='valid', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='valid', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='valid', one_hot=False)
        elif self.mode == "test":
            self.derm_data = derm_data.get_img_paths(data_type='test', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='test', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='test', one_hot=False)
        # print(f"train_idxs: {len(train_indexes)}")
        # print(f"valid_idxs: {len(valid_indexes)}")
        # print(f"len(self.derm_data): {len(self.derm_data)}")
        # print(f"len(self.clinic_data): {len(self.clinic_data)}")
        # print(f"len(self.labels): {len(self.labels)}")


class SevenPCBaseDataset2(SevenPCBaseDataset):
    def __getitem__(self, index):
        derm_image = self.derm_data[index]
        clinic_image = self.clinic_data[index]
        label = torch.as_tensor([self.labels[i].iloc[index] for i in self.__class__.LABEL_ORD])

        # derm_image = np.asarray(pil_image_loader(derm_image))[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        # clinic_image = np.asarray(load_img(clinic_image))[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        # derm_image = Image.fromarray(load_img(derm_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount])
        # clinic_image = Image.fromarray(load_img(clinic_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount])
        derm_image = load_img(derm_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        clinic_image = load_img(clinic_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        h1, w1, _ = derm_image.shape
        h2, w2, _ = clinic_image.shape
        h = min(h1, h2)
        w = min(w1, w2)
        derm_image = derm_image[:h, :w, :]
        clinic_image = clinic_image[:h, :w, :]
        derm_image = torch.tensor(derm_image).permute((2, 0, 1)).contiguous()
        clinic_image = torch.tensor(clinic_image).permute((2, 0, 1)).contiguous()
        images = torch.stack([derm_image, clinic_image], dim=0).to(dtype=torch.float32).div(255)

        if self.data_trans is None:
            derm_trans = derm_image
            clinic_trans = clinic_image
        else:
            image_trans = self.data_trans(images)
            derm_trans = image_trans[0]
            clinic_trans = image_trans[1]

        if self.return_index:
            return index, (derm_trans, clinic_trans, label)
        return (derm_trans, clinic_trans, label)


class SevenPCSwavDataset(Dataset):
    LABEL_ORD = ["DIAG", "PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]

    def __init__(
        self,
        args,
        mode,
        return_index=False,
    ):
        super().__init__()
        logger = logging.getLogger(f"{args.logger_name}.{self.__class__.__name__}")

        self.data_path = args.data_path + "/images"
        self.meta_dir = args.data_path
        self.mode = mode
        self.return_index = return_index

        # Apply cropping around the boundary of the images to get rid of the black boundary.
        self.crop_amount = 25
        self._prepare()

        logger.info(
            f"============ Building {self.__class__.__name__} ============\n"
            f"- dataset split: {mode}\n"
            f"- length: {self.__len__()}"
        )

        color_transform = [get_color_distortion(), transforms.GaussianBlur(19)]
        
        trans = []
        for i in range(len(args.size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                args.size_crops[i],
                scale=(args.min_scale_crops[i], args.max_scale_crops[i]),
                antialias=True,
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                # transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)])
            ] * args.nmb_crops[i])
        self.data_trans = trans

    def __getitem__(self, index):
        derm_image = self.derm_data[index]
        clinic_image = self.clinic_data[index]
        label = torch.as_tensor([self.labels[i].iloc[index] for i in self.__class__.LABEL_ORD])

        derm_image = load_img(derm_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        clinic_image = load_img(clinic_image)[self.crop_amount:-self.crop_amount, self.crop_amount:-self.crop_amount]
        h1, w1, _ = derm_image.shape
        h2, w2, _ = clinic_image.shape
        h = min(h1, h2)
        w = min(w1, w2)
        derm_image = derm_image[:h, :w, :]
        clinic_image = clinic_image[:h, :w, :]

        derm_image = torch.tensor(derm_image).permute((2, 0, 1)).contiguous()
        clinic_image = torch.tensor(clinic_image).permute((2, 0, 1)).contiguous()
        images = torch.stack([derm_image, clinic_image], dim=0).to(dtype=torch.float32).div(255)

        image_trans = list(map(lambda trans: trans(images), self.data_trans))
        derm_trans = [img[0] for img in image_trans]
        clinic_trans = [img[1] for img in image_trans]

        # derm_multi_crops = list(map(lambda trans: trans(derm_image), self.data_trans))
        # clinic_multi_crops = list(map(lambda trans: trans(clinic_image), self.data_trans))
        if self.return_index:
            return index, (derm_trans, clinic_trans)
        return (derm_trans, clinic_trans)

    def __len__(self):
        return len(self.derm_data)

    def _prepare(self):
        meta_df = pd.read_csv(os.path.join(self.meta_dir, "meta.csv"))
        train_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "train_indexes.csv"))["indexes"])
        valid_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "valid_indexes.csv"))["indexes"])
        test_indexes = list(pd.read_csv(os.path.join(self.meta_dir, "test_indexes.csv"))["indexes"])

        derm_data = SevenPCGroupDataset(dir_images=self.data_path, 
                    metadata_df=meta_df.copy(), # Copy as is modified.
                    train_indexes=train_indexes, valid_indexes=valid_indexes, 
                    test_indexes=test_indexes)

        # label is 1 diagnosis + 7 critiera: ["PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]
        if self.mode == "train":
            self.derm_data = derm_data.get_img_paths(data_type='train', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='train', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='train', one_hot=False)
        elif self.mode == "val":
            self.derm_data = derm_data.get_img_paths(data_type='valid', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='valid', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='valid', one_hot=False)
        elif self.mode == "test":
            self.derm_data = derm_data.get_img_paths(data_type='test', img_type='derm')
            self.clinic_data = derm_data.get_img_paths(data_type='test', img_type='clinic')
            self.labels = derm_data.get_labels(data_type='test', one_hot=False)


class ISIC17Dataset(Dataset):
    LABEL_ORD = ["pigment_network", "negative_network", "streaks", "milia_like_cyst"]

    def __init__(
        self,
        args,
        data_trans,
        mode,
        return_index=False,
    ):
        super().__init__()
        logger = logging.getLogger(f"{args.logger_name}.{self.__class__.__name__}")

        self.data_path = args.data_path
        self.data_trans = data_trans
        self.mode = mode
        self.return_index = return_index

        self._prepare()

        logger.info(
            f"============ Building {self.__class__.__name__} ============\n"
            f"- dataset split: {mode}\n"
            f"- length: {self.__len__()}"
        )

    def __getitem__(self, index):
        data = self.data[index]
        image_id = data[0]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.fromarray(load_img(image_path))
        label = torch.as_tensor(data[1:])


        if self.data_trans is None:
            image_trans = image
        else:
            image_trans = self.data_trans(image)

        if self.return_index:
            return index, (image_trans, label)
        return (image_trans, label)

    def __len__(self):
        return len(self.data)

    def _prepare(self):
        if self.mode == "train":
            self.image_dir = os.path.join(self.data_path, "ISIC-2017_Training_Data")
            self.label_csv = os.path.join(self.data_path, "train_labels.csv")
        elif self.mode == "test":
            self.image_dir = os.path.join(self.data_path, "ISIC-2017_Test_v2_Data")
            self.label_csv = os.path.join(self.data_path, "test_labels.csv")

        df = pd.read_csv(self.label_csv)
        # self.image_ids = df["image_id"].to_list()
        # self.labels = df[self.__class__.LABEL_ORD].to_numpy().tolist()
        self.data = df.to_numpy().tolist()


class ISIC18Dataset(Dataset):
    LABEL_ORD = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]

    def __init__(
        self,
        args,
        data_trans,
        mode,
        return_index=False,
    ):
        super().__init__()
        logger = logging.getLogger(f"{args.logger_name}.{self.__class__.__name__}")

        self.data_path = args.data_path
        self.data_trans = data_trans
        self.mode = mode
        self.return_index = return_index

        self._prepare()

        logger.info(
            f"============ Building {self.__class__.__name__} ============\n"
            f"- dataset split: {mode}\n"
            f"- length: {self.__len__()}"
        )

    def __getitem__(self, index):
        data = self.data[index]
        image_id = data[0]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.fromarray(load_img(image_path))
        label = torch.as_tensor(data[1:])


        if self.data_trans is None:
            image_trans = image
        else:
            image_trans = self.data_trans(image)

        if self.return_index:
            return index, (image_trans, label)
        return (image_trans, label)

    def __len__(self):
        return len(self.data)

    def _prepare(self):
        if self.mode == "train":
            self.image_dir = os.path.join(self.data_path, "ISIC2018_Task1-2_Training_Input")
            self.label_csv = os.path.join(self.data_path, "train_labels.csv")
        elif self.mode == "test":
            self.image_dir = os.path.join(self.data_path, "ISIC2018_Task1-2_Test_Input")
            self.label_csv = os.path.join(self.data_path, "test_labels.csv")

        df = pd.read_csv(self.label_csv)
        # self.image_ids = df["image_id"].to_list()
        # self.labels = df[self.__class__.LABEL_ORD].to_numpy().tolist()
        self.data = df.to_numpy().tolist()