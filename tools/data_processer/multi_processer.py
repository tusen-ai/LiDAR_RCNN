from multiprocessing import Process, Queue


class MultiProcesser(object):
    def __init__(self, flist, process_func, num_workers):
        assert type(flist) in (list, tuple)
        self.flist = flist
        self.process_func = process_func
        self.data_queue = Queue()
        self.result_queue = Queue(maxsize=1000)
        self.put_list()
        self.start_worker(num_workers)

    def put_list(self):
        for index in range(len(self.flist)):
            self.data_queue.put(self.flist[index])

    def get_result(self):
        while True:
            try:
                yield self.result_queue.get(timeout=30)
            except:
                return

    def start_worker(self, num_workers):
        workers = []
        def eval_worker(data_queue, result_queue):
            while True:
                output_dict = data_queue.get()
                for data in self.process_func(output_dict):
                    result_queue.put(data)
        for _ in range(num_workers):
            workers.append(Process(target=eval_worker, args=(self.data_queue, self.result_queue)))
        for w in workers:
            w.daemon = True
            w.start()

    def __len__(self):
        return len(self.flist)

    def run(self):
        print('Start processing!')
        for data in self.get_result():
            yield data
    
