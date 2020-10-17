from dataset.shapestack import ShapeStack
from dataset.cars_real_traffic import CarsRealTraffic
from dataset.clevr import CLEVR_v1
from dataset.clevr_obc import CLEVROBC
from dataset.dummy import Dummy


datasets = {'shapestacks': ShapeStack,
            'clevr-obc': CLEVROBC,
            'clevr': CLEVR_v1,
            'cars_real_traffic': CarsRealTraffic,
            'dummy': Dummy
            }


def init_dataset(dataset, data_root, eval_only=False, TRAIN={}, VAL={}):
    if eval_only:
        dataset = 'dummy'
        VAL = {'epoch_size': VAL.get('epoch_size', 10000),
               'image_size': TRAIN.get('image_size', 128),
               'seq_len': VAL.get('seq_len', 1)}
    try:
        datasets[dataset]
    except KeyError:
        raise KeyError('Unknown mode type: %s' % (dataset))

    training_set = None
    validation_set = None

    if TRAIN:
        training_set = datasets[dataset](data_root=data_root,
                                         train=True, **TRAIN)
    if VAL:
        validation_set = datasets[dataset](data_root=data_root,
                                           train=False, **VAL)

    return training_set, validation_set
