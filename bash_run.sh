nohup=false
source_path="tasks.csv"
target_path="main.py"
output_path=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--nohup)
            nohup_flag=true
            shift
            ;;
        -s|--source)
            source_path="$2"
            shift 2
            ;;
        -t|--target)
            target_path="$2"
            shift 2
            ;;
        -o|--output)
            output_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 读取Excel文件中的参数组合
IFS=$'\r\n' params=($(cat $source_path))

title_line=${params[0]}
IFS=',' read -ra titles <<< "$title_line"

# 遍历参数组合，并生成命令执行
for ((i = 1; i < ${#params[@]}; i++))
do
  param=${params[$i]}
  # 解析参数组合
  IFS=',' read -ra args <<< "$param"

  nohup=""
  if [ $nohup_flag ]; then
    nohup="nohup"
  fi
  command="$nohup python $target_path"

  for ((j = 0; j < ${#args[@]}; j++))
  do
    title=${titles[$j]}
    arg=${args[$j]}
    # 追加参数
    command="$command --$title $arg"
  done

  if [ $nohup_flag ] && [ ${#output_path} -gt 0 ] ; then
    command="$command > $output_path"
  fi
  
  echo $command
  eval $command &
done