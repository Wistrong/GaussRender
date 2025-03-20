from mmengine.utils import ManagerMixin
from torch.utils.tensorboard import SummaryWriter


class WrappedTBWriter(SummaryWriter, ManagerMixin):

    def __init__(self, name, **kwargs):
        SummaryWriter.__init__(self, **kwargs)
        ManagerMixin.__init__(self, name)
