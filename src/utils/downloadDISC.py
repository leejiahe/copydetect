import os
from multiprocessing.pool import ThreadPool

from torchvision.datasets.utils import download_and_extract_archive

ref_urls = [f'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_{i}.zip' 
            for i in range(20)]

train_urls = [f'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/train_{i}.zip' 
              for i in range(20)]

urls = ['https://dl.fbaipublicfiles.com/image_similarity_challenge/public/dev_queries.zip ',
        'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/final_queries.zip',
        ]



def download_file(urlInfo):
    download_root, url = urlInfo
    foldername = url.split('/')[-1]
    file_path = os.path.join(download_root, foldername)
    try:
        download_and_extract_archive(url = url,
                                    download_root = download_root,
                                    remove_finished = True)

    except:
        print(f'{foldername} cannot be downloaded')
        
    return foldername



def main(root_dir:str, threads:int = 8):
    disc_dir = os.path.join(root_dir, 'disc21')
    if not os.path.exists(disc_dir) :
        os.mkdir(disc_dir)

    ref_urls = [(disc_dir,
                 f'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_{i}.zip')
                for i in range(20)]
    
    train_urls = [(disc_dir,
                   f'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/train_{i}.zip')
                   for i in range(20)]
    
    val_url = [(disc_dir,
                'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/dev_queries.zip')]
    
    test_url = [(disc_dir,
                'https://dl.fbaipublicfiles.com/image_similarity_challenge/public/final_queries.zip')]
    

    urls = ref_urls + train_urls + val_url + test_url

    results = ThreadPool(threads).imap_unordered(download_file, urls)
    
    for result in results:
        print(f'Finish downloading {result}')
        
        

if __name__ == '__main__':
    main('/home/stevenlee/copydetect/data', threads = 8)