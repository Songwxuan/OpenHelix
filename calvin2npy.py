import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from tqdm import tqdm

import math
from typing import List, Any

def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    """将列表 lst 均分为 n 份，每份大小尽可能相等"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst: List[Any], n: int, k: int) -> List[Any]:
    """获取列表 lst 分成 n 份后的第 k 份"""
    chunks = split_list(lst, n)
    return chunks[k]

def factorize_calvin_data(task_desc, ts):
        sub_data = []
        import pdb; pdb.set_trace()
        for _ in range(ts[0], ts[1] + 1):
            if os.path.exists(os.path.join(calvin_data_path, f"episode_{str(_).zfill(7)}.npz")):
                ts_data = np.load(os.path.join(calvin_data_path, f"episode_{str(_).zfill(7)}.npz"))
                sub_data.append({
                    "task_desc": task_desc,
                    "image": ts_data["rgb_static"],
                    "action": ts_data["rel_actions"]
                })
        return sub_data

if __name__ == '__main__':
    # ========= Prepare CALVIN Dataset =========
    import pdb; pdb.set_trace()
    calvin_data_path = '/wangdonglin/calvin/task_ABC_D/training'

    language_ann_data = np.load(
        os.path.join(calvin_data_path, "lang_annotations", "auto_lang_ann.npy")
        , allow_pickle=True
    ).item()    

    ann = get_chunk(language_ann_data["language"]["ann"], n=10, k=8)
    indx = get_chunk(language_ann_data["info"]["indx"], n=10, k=8)

    with ThreadPoolExecutor(max_workers=1) as executor:
        data = list(
            tqdm(
                executor.map(
                    factorize_calvin_data,
                    ann,
                    indx
                ),
                total=len(indx),
                desc="Preprocess CALVIN ABC->D Training Data",
                ncols=100
            )
        )

    data = list(chain(*data))

    save_dir = os.path.join(os.getcwd(), "data_buffer")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"calvin_abc_d_train_8.npy"), data)


