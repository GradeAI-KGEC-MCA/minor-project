def select_directory_local():    
    import os
    ROOT = '/home/mav204/Documents/minor-project'
    os.chdir(ROOT)
    print(ROOT)

def select_directory_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir('/content/drive/MyDrive/minor-project')