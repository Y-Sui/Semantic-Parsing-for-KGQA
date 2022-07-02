#!/bin/bash
set -e
cd /hy-tmp
# 压缩包名称
file = "output.zip"
zip -q -r "${file}" output
# 通过oss上传到个人数据中的backup文件夹中
oss cp "${file}" oss://backup/
rm -f "${file}"

# 传输成功后关机
shutdown
