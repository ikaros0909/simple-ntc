import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')

print("반갑습니다.")
print("반갑습니다.")
print("반갑습니다.")