



var PDF_NAME = '<YOUR PDF FILE NAME>';
var BUCKET_NAME = '<YOUR BUCKET NAME>';
var SHEET_ID = '<YOUR SHEET ID>';


var pdfId = PDF_NAME.replace('.pdf', '').substr(0,4);
var bucketUrl = 'https://storage.googleapis.com/' + BUCKET_NAME + '/';
var imageUrl = bucketUrl + pdfId + '-images/%%page%%.png';

function getSheetApp() {
  return SpreadsheetApp.openById(SHEET_ID);
}

function getSheet() {
  return getSheetApp().getSheetByName(pdfId);
}

function getHeaderList() {
  var sheet = getSheet();
  return sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
}


function doGet(e) {
  var html = HtmlService.createTemplateFromFile('index');
  return html.evaluate();
}


function getImageUrl() {
  Logger.log('Image URL returned.');
  return imageUrl;
}

function downloadLabels() {

  
  var csv = '';  
  for (var i = 1; true; i += 100) {    
    
   
    var batchId = pdfId + '-' + ('000' + i).slice(-3);
    var url = bucketUrl + batchId + '-labels.csv';
    
    
    var resp = UrlFetchApp.fetch(url, {'muteHttpExceptions': true});
    if (resp.getResponseCode() == 200) {    
      csv += resp.getContentText('UTF-8');
    } else {
      break;
    }
  }
  

  var labelData = Utilities.parseCsv(csv);

  
  var oldSheet = getSheet();
  if (oldSheet) {
    oldSheet.setName(oldSheet.getName() + '.old.' + (new Date()).toLocaleTimeString())
  }

  var sheet = getSheetApp().insertSheet();
  sheet.setName(pdfId);
  sheet.insertRows(1, labelData.length);
  sheet.getRange(1, 1, labelData.length, labelData[0].length).setValues(labelData);
  Logger.log('Labels CSV downloaded.');
}


function updateLabel(id, label) {
  var finder = getSheet().createTextFinder(id);
  var idRange = finder.findNext();
  var idRow = idRange.getRow();
  var labelColumn = getHeaderList().indexOf('label') + 1;
  var labelRange = getSheet().getRange(idRow, labelColumn);
  labelRange.setValue(label);
  labelRange.setBackground('wheat');
}

function getParaDict() {
  
 
  if (getSheet() == null) {
    Logger.log('The sheet for ' + pdfId + ' is not available');
    return null;
  }

  var paraDict = buildParaDictFromSheet();
  
  
  Logger.log('paraDict returned.');
  return JSON.stringify(paraDict);
}

function buildParaDictFromSheet() {  
  
  var paraDict = new Object();
  var sheet = getSheet();
  var headerList = getHeaderList();
  var vals = sheet.getRange(2, 1, sheet.getLastRow(), sheet.getLastColumn()).getValues();
  
  var pageCount = 0;
  var paraCount = 0;
  vals.forEach(function(row) {
    
    
    var features = new Object();
    for (var i in row) {
      features[headerList[i]] = row[i];
    }
    
   
    var m = features.id.match(/(.*)-([0-9]+)-([0-9]+)/);
    if (!m) return;
    var pdfName = m[1];
    var page = m[2];
    var para = m[3];

    if (!paraDict[page]) {
      paraDict[page] = [];
      pageCount++;
    }
    paraDict[page].push(features);
    
    paraDict[page].sort(function (a, b) {
      return a.area < b.area ? 1 : (a.area == b.area ? 0 : -1);
    });
    paraCount++;
  });
  
  Logger.log('Built paraDict with ' + pageCount + ' pages, ' + paraCount + ' paragraphs.');
  return paraDict

}

