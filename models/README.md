# SAM2 模型文件存放目录

## 推荐存放位置

将下载的 SAM2 模型文件（`.pt` 文件）放在这个目录下：

```
surgerybot/
└── models/
    └── sam2/
        ├── sam2.1_hiera_base.pt    (推荐)
        ├── sam2.1_hiera_small.pt   (快速)
        ├── sam2.1_hiera_large.pt   (高精度)
        └── sam2.1_hiera_tiny.pt    (最快)
```

## 下载模型

### 方法 1：使用 curl（推荐）

```bash
# 进入 models/sam2 目录
cd /Users/rz/Desktop/Singapore/课程/me5400/me5400/surgerybot/models/sam2

# 下载 SAM2.1 Hiera Base（推荐）
curl -L -o sam2.1_hiera_base.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base.pt

# 或下载其他模型
curl -L -o sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 方法 2：使用浏览器下载

1. 打开下载链接（见下方）
2. 下载完成后，将 `.pt` 文件移动到 `models/sam2/` 目录

## 模型下载链接

- **SAM2.1 Hiera Base** (推荐，平衡): 
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base.pt

- **SAM2.1 Hiera Small** (快速): 
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

- **SAM2.1 Hiera Large** (高精度): 
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

- **SAM2.1 Hiera Tiny** (最快): 
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

## 使用说明

1. 将模型文件放在 `models/sam2/` 目录下
2. 启动程序
3. 点击 "Load SAM2 Model" 按钮
4. 文件选择对话框会自动打开到 `models` 目录
5. 选择对应的 `.pt` 文件即可

## 注意事项

- 模型文件较大（100MB - 1.5GB），下载需要时间
- 确保有足够的磁盘空间
- 模型文件只需下载一次，可以重复使用

