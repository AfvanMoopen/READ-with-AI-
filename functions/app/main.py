

import base64
import os
import re
import csv
import io
import tempfile
import ghostscript
import locale
import glob
import time

from pydub import AudioSegment

from google.cloud import storage
from google.cloud import vision
from google.cloud import texttospeech
from google.cloud import automl_v1beta1 as automl
from google.protobuf import json_format

ANNOTATION_MODE = False


compute_region = "us-central1"
model_display_name = "<YOUR MODEL DISPLAY NAME>"
#
SECTION_BREAK = 2  # sec
CAPTION_BREAK = 1.5  # sec

LABEL_BODY = "body"
LABEL_HEADER = "header"
LABEL_CAPTION = "caption"
LABEL_OTHER = "other"
FEATURE_CSV_HEADER = (
    "id,text,chars,width,height,area,char_size,pos_x,pos_y,aspect,layout"
)


project_id = os.environ["GCP_PROJECT"]
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()
speech_client = texttospeech.TextToSpeechClient()
automl_client = automl.TablesClient(project=project_id, region=compute_region)


def p2a_gcs_trigger(file, context):

 
    file_name = file["name"]
    bucket = None
    file_blob = None
    while bucket == None or file_blob == None:  
        bucket = storage_client.get_bucket(file["bucket"])
        file_blob = bucket.get_blob(file_name)
        time.sleep(1)

 
    if file_name.lower().endswith(".pdf"):
        p2a_ocr_pdf(bucket, file_blob)
        return

   
    if file_name.lower().endswith(".json"):
        p2a_predict(bucket, file_blob)
        return

    if file_name.lower().endswith("tables_1.csv"):
        if ANNOTATION_MODE:
            p2a_generate_labels(bucket, file_blob)
        else:
            p2a_generate_speech(bucket, file_blob)
        return


def p2a_ocr_pdf(bucket, pdf_blob):

    
    gcs_source_uri = "gs://{}/{}".format(bucket.name, pdf_blob.name)
    gcs_source = vision.types.GcsSource(uri=gcs_source_uri)
    feature = vision.types.Feature(
        type=vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION
    )
    input_config = vision.types.InputConfig(
        gcs_source=gcs_source, mime_type="application/pdf"
    )

   
    pdf_id = pdf_blob.name.replace(".pdf", "")[:4]  
    gcs_dest_uri = "gs://{}/{}".format(bucket.name, pdf_id + ".")
    gcs_destination = vision.types.GcsDestination(uri=gcs_dest_uri)
    output_config = vision.types.OutputConfig(
        gcs_destination=gcs_destination, batch_size=100
    )

    async_request = vision.types.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )
    async_response = vision_client.async_batch_annotate_files(requests=[async_request])
    print("Started OCR for file {}".format(pdf_blob.name))

   
    if ANNOTATION_MODE:
        convert_pdf2png(bucket, pdf_blob)


def p2a_predict(bucket, json_blob):


    m = re.match("(.*).output-([0-9]+)-.*", json_blob.name)
    pdf_id = m.group(1)
    first_page = int(m.group(2))

   
    csv = FEATURE_CSV_HEADER + "\n"
    csv += build_feature_csv(json_blob, pdf_id, first_page)

   
    feature_file_name = "{}-{:03}-features.csv".format(pdf_id, first_page)
    feature_blob = bucket.blob(feature_file_name)
    feature_blob.upload_from_string(csv)
    print("Feature CSV file saved: {}".format(feature_file_name))
    json_blob.delete()

    
    gcs_input_uris = ["gs://{}/{}".format(bucket.name, feature_file_name)]
    gcs_output_uri = "gs://{}".format(bucket.name)

  
    print("Started AutoML batch prediction for {}".format(feature_file_name))
    response = automl_client.batch_predict(
        gcs_input_uris=gcs_input_uris,
        gcs_output_uri_prefix=gcs_output_uri,
        model_display_name=model_display_name,
    )
    response.result()
    print("Ended AutoML batch prediction for {}".format(feature_file_name))
    if not ANNOTATION_MODE:
        feature_blob.delete()


def build_feature_csv(json_blob, pdf_id, first_page):

  
    json_string = json_blob.download_as_string()
    json_response = json_format.Parse(json_string, vision.types.AnnotateFileResponse())

    csv = ""
    page_count = first_page
    for resp in json_response.responses:
        para_count = 0
        for page in resp.full_text_annotation.pages:

      
            page_features = []
            for block in page.blocks:
                if str(block.block_type) != "1":  
                    continue
                for para in block.paragraphs:
                    para_id = "{}-{:03}-{:03}".format(pdf_id, page_count, para_count)
                    f = extract_paragraph_feature(para_id, para)
                    page_features.append(f)
                    para_count += 1

         
            for f in page_features:
                csv += '{},"{}",{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n'.format(
                    f["para_id"],
                    f["text"],
                    f["chars"],
                    f["width"],
                    f["height"],
                    f["area"],
                    f["char_size"],
                    f["pos_x"],
                    f["pos_y"],
                    f["aspect"],
                    f["layout"],
                )

        page_count += 1
    return csv


def extract_paragraph_feature(para_id, para):

  
    text = ""
    for word in para.words:
        for symbol in word.symbols:
            text += symbol.text
            if hasattr(symbol.property, "detected_break"):
                break_type = symbol.property.detected_break.type
                if str(break_type) == "1":
                    text += " "  


    text = text.replace('"', "")


    text = re.sub("https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)

  
    x_list = []
    y_list = []
    for v in para.bounding_box.normalized_vertices:
        x_list.append(v.x)
        y_list.append(v.y)
    f = {}
    f["para_id"] = para_id
    f["text"] = text
    f["width"] = max(x_list) - min(x_list)
    f["height"] = max(y_list) - min(y_list)
    f["area"] = f["width"] * f["height"]
    f["chars"] = len(text)
    f["char_size"] = f["area"] / f["chars"] if f["chars"] > 0 else 0
    f["pos_x"] = (f["width"] / 2.0) + min(x_list)
    f["pos_y"] = (f["height"] / 2.0) + min(y_list)
    f["aspect"] = f["width"] / f["height"] if f["height"] > 0 else 0
    f["layout"] = "h" if f["aspect"] > 1 else "v"

    return f


def p2a_generate_speech(bucket, csv_blob):

    batch_id, sorted_ids, text_dict, label_dict = parse_prediction_results(
        bucket, csv_blob
    )

    mp3_blob_list = generate_mp3_files(bucket, sorted_ids, text_dict, label_dict)

    merge_mp3_files(bucket, batch_id, mp3_blob_list)


    folder_name = re.sub("/.*.csv", "", csv_blob.name)
    folder_blobs = storage_client.list_blobs(bucket, prefix=folder_name)
    for b in folder_blobs:
        b.delete()


def parse_prediction_results(bucket, csv_blob):
   
    csv_string = csv_blob.download_as_string().decode("utf-8")
    csv_file = io.StringIO(csv_string)
    reader = csv.DictReader(csv_file)
    text_dict = {}
    label_dict = {}
    for row in reader:

        id = row["id"]
        text = row["text"]
        text = text.replace("<", "")  
        text_dict[id] = text

    
        sc_other = float(row["label_other_score"])
        sc_body = float(row["label_body_score"])
        sc_caption = float(row["label_caption_score"])
        sc_header = float(row["label_header_score"])
        #        if sc_other > 0.7:
        if sc_other > max(sc_header, sc_body, sc_caption):
            label_dict[id] = LABEL_OTHER
        elif sc_header > max(sc_body, sc_caption):
            label_dict[id] = LABEL_HEADER
        elif sc_caption > sc_body:
            label_dict[id] = LABEL_CAPTION
        else:
            label_dict[id] = LABEL_BODY
    sorted_ids = sorted(text_dict.keys())
    first_id = sorted_ids[0]

    others = ""
    for id in sorted_ids:
        if label_dict[id] == LABEL_OTHER:
            others += text_dict[id] + " "
            sorted_ids.remove(id)


    last_id = None
    period_pattern = re.compile(r"^.*[.。」）)”]$")
    for id in sorted_ids:
        if last_id:
            is_bodypairs = (
                label_dict[id] == LABEL_BODY and label_dict[last_id] == LABEL_BODY
            )
            is_captpairs = (
                label_dict[id] == LABEL_CAPTION and label_dict[last_id] == LABEL_CAPTION
            )
            is_lastbody_nopediod = not period_pattern.match(text_dict[last_id])
            if (is_bodypairs and is_lastbody_nopediod) or is_captpairs:
                text_dict[id] = text_dict[last_id] + text_dict[id]
                sorted_ids.remove(last_id)
        last_id = id

    m = re.match("(.*-[0-9]+)-.*", first_id)
    batch_id = m.group(1)

    return batch_id, sorted_ids, text_dict, label_dict


def generate_mp3_files(bucket, sorted_ids, text_dict, label_dict):

    ssml = ""
    section_break = '<break time="{}s"/>'.format(SECTION_BREAK)
    caption_break = '<break time="{}s"/>'.format(CAPTION_BREAK)
    mp3_blob_list = []
    prev_id = None
    for id in sorted_ids:

        
        if len(ssml) + len(text_dict[id]) > 4500:
            mp3_blob = generate_mp3_for_ssml(bucket, prev_id, ssml)
            mp3_blob_list.append(mp3_blob)
            ssml = ""

       
        if label_dict[id] == LABEL_BODY:
            ssml += "<p>" + text_dict[id] + "</p>\n"
        elif label_dict[id] == LABEL_CAPTION:
            ssml += caption_break + text_dict[id] + caption_break + "\n"
        elif label_dict[id] == LABEL_HEADER:
            ssml += section_break + text_dict[id] + section_break + "\n"

        prev_id = id

    mp3_blob = generate_mp3_for_ssml(bucket, prev_id, ssml)
    mp3_blob_list.append(mp3_blob)
    return mp3_blob_list


def generate_mp3_for_ssml(bucket, id, ssml):

    ssml = "<speak>\n" + ssml + "</speak>\n"
    synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml)
    voice = texttospeech.types.VoiceSelectionParams(
        language_code="ja-JP", ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3, speaking_rate=1.5
    )

   
    try:
        response = speech_client.synthesize_speech(synthesis_input, voice, audio_config)
    except Exception as e:
        print("Retrying speech generation...") 
        response = speech_client.synthesize_speech(synthesis_input, voice, audio_config)

    mp3_file_name = id + ".mp3"
    mp3_blob = bucket.blob(mp3_file_name)
    mp3_blob.upload_from_string(response.audio_content, content_type="audio/mpeg")
    print("MP3 file saved: {}".format(mp3_file_name))
    return mp3_blob


def merge_mp3_files(bucket, batch_id, mp3_blob_list):

    
    print("Started merging mp3 files for {}".format(batch_id))
    merged_mp3 = None
    for mp3_blob in mp3_blob_list:
        mp3_file = io.BytesIO(mp3_blob.download_as_string())
        mp3_data = AudioSegment.from_file(mp3_file, format="mp3")
        if merged_mp3:
            merged_mp3 += mp3_data
        else:
            merged_mp3 = mp3_data

    
    merged_mp3_file_name = (
        re.sub("[0-9][0-9]$", "", batch_id) + ".mp3"
    )  
    merged_mp3_file_path = tempfile.gettempdir() + "/" + merged_mp3_file_name
    merged_mp3.export(merged_mp3_file_path, format="mp3")
    merged_mp3_blob = bucket.blob(merged_mp3_file_name)
    merged_mp3_blob.upload_from_filename(
        merged_mp3_file_path, content_type="audio/mpeg"
    )

   
    bucket.delete_blobs(mp3_blob_list)
    print("Ended merging mp3 files: {}".format(merged_mp3_file_name))



def p2a_generate_labels(bucket, automl_csv_blob):

   
    batch_id, sorted_ids, text_dict, label_dict = parse_prediction_results(
        bucket, automl_csv_blob
    )

  
    features_blob = bucket.get_blob(batch_id + "-features.csv")
    features_string = features_blob.download_as_string().decode("utf-8")
    csv = ""
    if batch_id.endswith("001"):
        csv += FEATURE_CSV_HEADER + ",label\n"
    for l in features_string.split("\n"):
        m = re.match("^([^,]*-[0-9]+-[0-9]+),.*$", l)
        if m:
            id = m.group(1)
            label = label_dict[id]
            csv += l + "," + label + "\n"

    labels_file_name = batch_id + "-labels.csv"
    labels_blob = bucket.blob(labels_file_name)
    labels_blob.upload_from_string(csv)
    labels_blob.make_public()
    print("Predicted results saved: {}".format(labels_file_name))
    features_blob.delete()

   
    folder_name = re.sub("/.*.csv", "", automl_csv_blob.name)
    folder_blobs = storage_client.list_blobs(bucket, prefix=folder_name)
    for b in folder_blobs:
        b.delete()


def convert_pdf2png(bucket, pdf_blob):

    print("Downloading PDF: {}".format(pdf_blob.name))
    _, pdf_file_name = tempfile.mkstemp()
    with open(pdf_file_name, "w+b") as pdf_file:
        pdf_blob.download_to_file(pdf_file)

   
    print("Converting PDF to PNGs for {}".format(pdf_blob.name))
    pdf_prefix = pdf_blob.name.replace(".pdf", "")[:4]
    png_tempdir = tempfile.mkdtemp()
    args = [
        "pdf2png",
        "-dSAFER",
        "-sDEVICE=pngalpha",
        "-r100",
        "-sOutputFile={}/%03d.png".format(png_tempdir),
        pdf_file_name,
    ]
    encoding = locale.getpreferredencoding()
    args = [a.encode(encoding) for a in args]
    ghostscript.Ghostscript(*args)

 
    print("Saving PNGs for {}".format(pdf_blob.name))
    for f in glob.glob(png_tempdir + "/*"):
        png_blob = bucket.blob(pdf_prefix + "-images/" + os.path.split(f)[1])
        png_blob.upload_from_filename(f, content_type="image/png")
        png_blob.make_public()
        os.remove(f)
    print("Ended converting PDF to PNGs for {}".format(pdf_blob.name))
    os.remove(pdf_file_name)
