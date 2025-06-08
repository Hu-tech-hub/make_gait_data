# ======================
# 3. main.py - 프로그램 진입점
# ======================
"""
main.py - 보행 분석 시스템 메인 진입점
"""

import sys
from PyQt5.QtWidgets import QApplication
from gait_analyzer_gui import GaitAnalyzerGUI


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성 및 표시
    window = GaitAnalyzerGUI()
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
