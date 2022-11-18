import os
from multiprocessing.pool import ThreadPool

from torchvision.datasets.utils import download_and_extract_archive

def download_file(urlInfo):
    download_root, url = urlInfo
    foldername = url.split('/')[-1]
    download_and_extract_archive(url = url,
                                 download_root = download_root,
                                 remove_finished = True)
    return foldername

def main(root_dir):
    copy_dir = os.path.join(root_dir, 'copydays')
    if not os.path.exists(copy_dir) :
        os.mkdir(copy_dir)
    
    urls = {'original':'http://pascal.inrialpes.fr/data2/holidays/copydays_original.tar.gz',
            'crop':'http://pascal.inrialpes.fr/data2/holidays/copydays_crop.tar.gz',
            'jpegqual':'http://pascal.inrialpes.fr/data2/holidays/copydays_jpeg.tar.gz',
            'strong':'http://pascal.inrialpes.fr/data2/holidays/copydays_strong.tar.gz',
            }
    allurlInfo = []
    for name, url in urls.items():
        folder_path = os.path.join(copy_dir, name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            
        allurlInfo.append((folder_path, url))
    
    results = ThreadPool(4).imap_unordered(download_file, allurlInfo)
    
    for result in results:
        print(f'Finish downloading {result}')

if __name__ == '__main__':
    main('/home/stevenlee/copydetect/data')