import pandas as pd
import csv
import requests

# API 金鑰與 API URL 配置
GEMINI_API_KEY = "AIzaSyD_bkUIDtfzwqU5O4lgbP_9ugOrS9iZaqw"
API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'


# 呼叫 API 分析案件內容
def analyze_content_with_api(content, retry=3):
    """
    使用 API 分析內容，判斷案件類型。
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": (
                f"以下是一段契約內容，請協助判斷其中內容是**懲罰性編號1**還是**損害賠償性編號2**，\n\n"
                f"分析依據：\n"
                f"**懲罰性編號1**\n"
                f"1. 為了對違約方進行懲罰，通常超過實際損失的範圍，並無與具體損失掛鉤。\n"
                f"2. 提及額外罰金，且這些罰金無法與實際損失相對應，則屬於懲罰性違約金。\n"
                f"3. 金額遠超過實際損害的合理比例（通常超過損害的30%），此類設定是對違約行為的額外懲罰，而非對實際損失的補償。\n"
                f"4. 無法直接與實際損失或費用相連結，金額通常是預設的固定數額或按契約條款預定的比例計算。\n"
                f"5. 伴隨強制執行條款，用以迫使違約方履行契約或承擔額外的經濟責任。\n\n"
                f"**損害賠償性編號2**\n"
                f"1. 補償由違約行為所造成的實際損失。其金額通常不超過損失的30%，以合理比例反映實際損失。\n"
                f"2. 返還款項、不當得利返還或程序費用，且未提及額外罰金或懲罰性條款。\n"
                f"3. 基於實際損害設定，金額通常不會超過損害的30%。若金額過高，則需進一步檢視其是否合理。\n"
                f"4. 若契約中涉及雙方協商的和解金額，且金額與實際損失相符，則屬於損害賠償性。\n"
                f"5. 若金額設定目的是賠償由違約行為引起的實際損失，而非對違約方進行懲罰。\n\n"
                f"契約內容若無法確定，請務必選擇最接近的分類：**懲罰性編號1**或**損害賠償性編號2**，不允許未知分類。\n\n"
                f"內容如下：\n{content}"
            )}]
        }]
    }
    try:
        response = requests.post(API_URL, params={'key': GEMINI_API_KEY}, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        return "**損害賠償性(編號2)**"
    except requests.exceptions.RequestException as e:
        if retry > 0:
            print(f"API 呼叫失敗: {e}，嘗試重新呼叫...")
            return analyze_content_with_api(content, retry - 1)
        print(f"API 呼叫多次失敗: {e}")
        return "**損害賠償性(編號2)**"


# 將案件類型映射為數字代碼
def map_case_type_to_code(case_type):
    """
    將案件類型文字描述轉換為對應的數字代碼：
    **損害賠償性編號2** => 2, **懲罰性編號1** => 1
    """
    if ("損害賠償性編號2**" in case_type or
            "損害賠償性編號2" in case_type or
            "編號2：損害賠償性" in case_type or
            "編號2 損害賠償性" in case_type or
            "損害賠償性違約金（編號2）" in case_type or
            "損害賠償性違約金(編號2)" in case_type or
            "損害賠償性違約金編號2" in case_type or
            "損害賠償性 (編號2)" in case_type or
            "損害賠償性（編號2）" in case_type):
        return 2
    elif ("懲罰性編號1**" in case_type or
          "懲罰性編號1" in case_type or
          "編號1：懲罰性" in case_type or
          "編號1 懲罰性" in case_type or
          "懲罰性違約金（編號1）" in case_type or
            "懲罰性違約金(編號1)" in case_type or
            "懲罰性違約金編號1" in case_type or
            "懲罰性 (編號1)" in case_type or
            "懲罰性（編號1）" in case_type):
        return 1
    return 0


# 主函式
def main():
    """
    主函式，讀取檔案並分析判決書全文內容。
    """
    input_file = "judgment_data_2.csv"
    output_file = "judgment_data_analysis.csv"
    target_file = "Target.csv"

    # 讀取資料
    data = pd.read_csv(input_file)

    # 初始化輸出文件
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['序號', '案件名稱', '最終違約金類型', '案件類型數字'])
        writer.writeheader()

    with open(target_file, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Target'])

    # 分析每個案件內容
    for index, row in data.iterrows():
        full_content = row['FullContent']
        case_name = row['案件名稱'] if '案件名稱' in row else f"案件_{index + 1}"

        # 呼叫 API 進行分析
        final_type = analyze_content_with_api(full_content)
        case_type_code = map_case_type_to_code(final_type)

        # 寫入分析結果
        case_data = {
            '序號': index + 1,
            '案件名稱': case_name,
            '最終違約金類型': final_type,
            '案件類型數字': case_type_code
        }

        with open(output_file, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=['序號', '案件名稱', '最終違約金類型', '案件類型數字'])
            writer.writerow(case_data)

        with open(target_file, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([case_type_code])


if __name__ == '__main__':
    main()
