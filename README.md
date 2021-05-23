# NER_competition

標記類型有 : 
名字（name）
地點（location）
時間（time）
聯絡方式（contact）
編號（id）
職業（profession）
個人生物標誌（biomarker）
家庭成員（family）
有名的臨床事件（clinical_event）
特殊專業或技能（special_skills）
獨家或聞名的治療方法（unique_treatment）
帳號（account）
所屬團體（organization）
就學經歷或學歷（education）
金額（money）
所屬品的特殊標誌（belonging_mark）
報告數值（med_exam）
其他（others）

模型設計 : 
1.透過標點符號切割將句子長度控制在50字內  
2.bert(不做pre-train)做word embedding產生word vector後  
3.使用Bi-LSTM -> CRF來標記  
  (Bi-LSTM 140 units, 50% drop-out, learning-rate : 1e-4)


最終結果 :

                precision    recall  f1-score   support

      med_exam     0.7333    0.7196    0.7264       107
          time     0.7079    0.7858    0.7448       663
      location     0.8148    0.8800    0.8462        50
    profession     0.0000    0.0000    0.0000        19
          name     0.9333    0.7368    0.8235        57
         money     0.4167    0.6818    0.5172        22
       contact     0.5000    0.5556    0.5263         9
        others     0.0000    0.0000    0.0000         3
            ID     0.0000    0.0000    0.0000         2
        family     0.5000    0.5000    0.5000         6
     education     0.0000    0.0000    0.0000         2

     micro avg     0.7077    0.7521    0.7292       940
     macro avg     0.7004    0.7521    0.7233       940
     
資料來源 : Aidea競賽方提供，因此未上傳資料集。 
