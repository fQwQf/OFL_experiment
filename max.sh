for f in *.log; do                                                                                                ─╯
  echo -n "📄 文件 $f 最大值: "
  grep "The test accuracy of" "$f" | awk '{print $NF}' | sort -nr | head -n 1
done
