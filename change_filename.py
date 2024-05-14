import os

# 해당 폴더로 이동
os.chdir('/path/to/your/folder')

prev_name=input("변경 전 사진의 날짜")
curr_name=input("변경 후 사진의 날짜")
count = int(input("시작 번호 (덮어쓰기 주의하기!)"))

# 숫자로 된 파일 이름을 순회하면서 변경
for filename in os.listdir('.'):
    if filename.startswith('{}_'.format(prev_name)):
        newname = f'{curr_name}_{count}'
        os.rename(filename, newname)
        count += 1