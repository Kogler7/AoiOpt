# 检查参数数量
if [ $# -ne 2 ]; then
    echo "请提供下界和上界作为参数"
    exit 1
fi

# 提取下界和上界
lower=$1
upper=$2

# 检查下界和上界是否为数字
if ! [[ $lower =~ ^[0-9]+$ && $upper =~ ^[0-9]+$ ]]; then
    echo "下界和上界必须为非负整数"
    exit 1
fi

# 检查下界是否小于等于上界
if [ $lower -gt $upper ]; then
    echo "下界不能大于上界"
    exit 1
fi

# 杀死进程
ps -ef | awk -v lower=$lower -v upper=$upper '$2 >= lower && $2 <= upper { print $2 }' | xargs kill