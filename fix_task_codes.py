#!/usr/bin/env python3
"""
fix_task_codes.py - T03/T04 코드 수정 스크립트

batch_gait_analyzer.py의 올바른 매핑에 따라:
- pain_gait -> T04 (현재 T03으로 잘못 저장됨)
- hemiparetic_gait -> T03 (현재 T04로 잘못 저장됨)

이 스크립트는 파일명의 T03과 T04를 서로 바꿔줍니다.
"""

import os
import shutil
from pathlib import Path


def fix_task_codes_in_directory(directory_path: str, dry_run: bool = True):
    """
    디렉토리의 파일들에서 T03과 T04 코드를 서로 바꿔줍니다.
    
    Args:
        directory_path (str): 수정할 디렉토리 경로
        dry_run (bool): True면 실제 변경하지 않고 미리보기만 출력
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {directory_path}")
        return
    
    # 수정할 파일들 찾기
    t03_files = list(directory.glob("*T03R*_support_labels.csv"))
    t04_files = list(directory.glob("*T04R*_support_labels.csv"))
    
    print(f"\n📁 처리 디렉토리: {directory_path}")
    print(f"🔍 T03 파일 개수: {len(t03_files)}")
    print(f"🔍 T04 파일 개수: {len(t04_files)}")
    
    if not t03_files and not t04_files:
        print("⚠️  T03 또는 T04 파일을 찾을 수 없습니다.")
        return
    
    changes = []
    
    # T03 -> T04로 변경할 파일들
    for file_path in t03_files:
        old_name = file_path.name
        new_name = old_name.replace("T03", "T04")
        new_path = file_path.parent / new_name
        changes.append((file_path, new_path, "T03→T04"))
    
    # T04 -> T03으로 변경할 파일들  
    for file_path in t04_files:
        old_name = file_path.name
        new_name = old_name.replace("T04", "T03")
        new_path = file_path.parent / new_name
        changes.append((file_path, new_path, "T04→T03"))
    
    if not changes:
        print("✅ 변경할 파일이 없습니다.")
        return
    
    # 변경 사항 출력
    print(f"\n📋 변경 예정 파일 목록 ({len(changes)}개):")
    print("-" * 80)
    for old_path, new_path, change_type in changes:
        print(f"{change_type}: {old_path.name} → {new_path.name}")
    
    if dry_run:
        print("\n🔍 [미리보기 모드] 실제 변경하려면 dry_run=False로 실행하세요.")
        return
    
    # 실제 파일명 변경 (안전한 3단계 방식)
    print(f"\n🚀 파일명 변경 시작 (3단계 안전 방식)...")
    success_count = 0
    error_count = 0
    
    # 1단계: T03 파일들을 T03_로 임시 변경
    print("1단계: T03 → T03_ (임시)")
    t03_temp_files = []
    for old_path, new_path, change_type in changes:
        if "T03→T04" in change_type:
            try:
                temp_name = old_path.name.replace("T03", "T03_")
                temp_path = old_path.parent / temp_name
                shutil.move(str(old_path), str(temp_path))
                t03_temp_files.append((temp_path, new_path))
                print(f"  ✅ {old_path.name} → {temp_name}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ 오류 - {old_path.name}: {str(e)}")
                error_count += 1
    
    # 2단계: T04 파일들을 T03으로 변경
    print("\n2단계: T04 → T03")
    for old_path, new_path, change_type in changes:
        if "T04→T03" in change_type:
            try:
                # 대상 파일 존재 여부 확인
                if new_path.exists():
                    print(f"  ⚠️ 대상 파일 이미 존재: {new_path.name}")
                    error_count += 1
                    continue
                    
                shutil.move(str(old_path), str(new_path))
                print(f"  ✅ {old_path.name} → {new_path.name}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ 오류 - {old_path.name}: {str(e)}")
                error_count += 1
    
    # 3단계: T03_ 파일들을 T04로 변경
    print("\n3단계: T03_ → T04")
    for temp_path, final_path in t03_temp_files:
        try:
            # 대상 파일 존재 여부 확인
            if final_path.exists():
                print(f"  ⚠️ 대상 파일 이미 존재: {final_path.name}")
                error_count += 1
                continue
                
            shutil.move(str(temp_path), str(final_path))
            print(f"  ✅ {temp_path.name} → {final_path.name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 오류 - {temp_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\n📊 결과: 성공 {success_count}개, 실패 {error_count}개")


def fix_all_subjects(base_path: str = "support_label_data", dry_run: bool = True):
    """
    모든 피험자 폴더에서 T03/T04 코드를 수정합니다.
    
    Args:
        base_path (str): support_label_data 경로
        dry_run (bool): True면 미리보기만 출력
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {base_path}")
        return
    
    print("=" * 80)
    print("🔧 T03/T04 Task Code 수정 스크립트")
    print("=" * 80)
    print("📝 올바른 매핑 (batch_gait_analyzer.py 기준):")
    print("   pain_gait → T04")
    print("   hemiparetic_gait → T03")
    print()
    print("🔄 변경 작업:")
    print("   현재 T03 파일들 → T04로 변경 (pain_gait 파일들)")
    print("   현재 T04 파일들 → T03으로 변경 (hemiparetic_gait 파일들)")
    print("=" * 80)
    
    # 피험자별 폴더 처리
    subject_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('SA')]
    subject_dirs.sort()
    
    if not subject_dirs:
        print("⚠️  SA로 시작하는 피험자 폴더를 찾을 수 없습니다.")
        return
    
    total_success = 0
    total_files = 0
    
    for subject_dir in subject_dirs:
        print(f"\n👤 피험자: {subject_dir.name}")
        
        # 해당 피험자 디렉토리의 파일들 처리
        t03_files = list(subject_dir.glob("*T03R*_support_labels.csv"))
        t04_files = list(subject_dir.glob("*T04R*_support_labels.csv"))
        
        subject_total = len(t03_files) + len(t04_files)
        total_files += subject_total
        
        if subject_total == 0:
            print("   ⚠️  T03/T04 파일 없음")
            continue
            
        fix_task_codes_in_directory(str(subject_dir), dry_run)
        
        if not dry_run:
            total_success += subject_total
    
    print("\n" + "=" * 80)
    print(f"🎯 전체 요약:")
    print(f"   📁 처리된 피험자: {len(subject_dirs)}명")
    print(f"   📄 총 파일 수: {total_files}개")
    
    if dry_run:
        print(f"   🔍 [미리보기 모드] 실제 변경하려면 실행 명령을 확인하세요.")
    else:
        print(f"   ✅ 변경 완료: {total_success}개")
    
    print("=" * 80)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="T03/T04 Task Code 수정 스크립트")
    parser.add_argument("--path", default="support_label_data", 
                       help="수정할 디렉토리 경로 (기본값: support_label_data)")
    parser.add_argument("--execute", action="store_true",
                       help="실제로 파일명을 변경합니다 (기본값: 미리보기만)")
    parser.add_argument("--subject", type=str,
                       help="특정 피험자만 처리 (예: SA01)")
    
    args = parser.parse_args()
    
    # 현재 디렉토리 확인
    if not os.path.exists(args.path):
        print(f"❌ 경로를 찾을 수 없습니다: {args.path}")
        print("   현재 디렉토리에서 실행해주세요.")
        return
    
    dry_run = not args.execute
    
    if args.subject:
        # 특정 피험자만 처리
        subject_path = os.path.join(args.path, args.subject)
        if not os.path.exists(subject_path):
            print(f"❌ 피험자 폴더를 찾을 수 없습니다: {subject_path}")
            return
        
        print(f"👤 피험자 {args.subject} 처리 중...")
        fix_task_codes_in_directory(subject_path, dry_run)
    else:
        # 모든 피험자 처리
        fix_all_subjects(args.path, dry_run)
    
    if dry_run:
        print("\n" + "🔍" * 20)
        print("실제 변경하려면 다음 명령어를 사용하세요:")
        print(f"python fix_task_codes.py --execute")
        if args.subject:
            print(f"python fix_task_codes.py --execute --subject {args.subject}")


if __name__ == "__main__":
    main() 