import lmdb

class Dataset(object):
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class LMDBIterator(object):
    def __init__(self, cur, transform, proto):
        self.cur = cur
        self.transform = transform
        self.has_next = self.cur.next()
        self.proto = proto

    def parse_byte(self, byte_str):
        tensor_protos = self.proto()
        tensor_protos.ParseFromString(byte_str)

        return self.transform(tensor_protos)

    def __next__(self):
        if not self.has_next:
            raise StopIteration
        try:
            item = self.parse_byte(self.cur.value())
            self.has_next = self.cur.next()
            return item
        except Exception as e:
            print(e)
            raise StopIteration

    def __iter__(self):
        return self


class LMDBDataset(Dataset):
    def __init__(self, db_path, transform, proto):
        self.db = lmdb.open(db_path, readonly=True)
        self.txn = self.db.begin()
        self.length = self.txn.stat()['entries']
        self.cur = self.txn.cursor()

        self.transform = transform

        self.proto = proto

    def __iter__(self):
        return LMDBIterator(self.cur, self.transform, self.proto)

    def __len__(self):
        return self.length

