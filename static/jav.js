
function isChecked(value){
    if(value.is(':checked')){
        return true;
    }
    return false;
}

function setSpinner(flag){
    var btn_area = $('#btn');
    var text = $('#btn span');
    var spinner = $('#btn div');

    if(flag){
        text.css('display', 'none');
        spinner.css('display', 'inline-block');
        btn_area.css('pointer-events', 'none');
    }
    else{
        text.css('display', 'inline-block');
        spinner.css('display', 'none');
        btn_area.css('pointer-events', 'auto');
    }
}

$('#btn').click(function(){
   var isRealtime = $('#realtime');
   var isReid = $('#reid');
   var isHeatmap = $('#heatmap');
   var yolo = $('#yolo option:selected').val();
   var reid_model = $('#reid_model option:selected').val();
   var ds_model = $('#ds_model option:selected').val();
   var frame_skip = $('#frame_skip input').val();
   var video_length = $('#video_length input').val();
   var heatmap_cumul_sec = $('#heatmap_cumul_sec input').val();
   var fps = $('#fps input').val();
   var videos_num = $('#videos_num input').val();
   var resol = $('#resol input').val();

   var param = {
       'realtime' : isChecked(isRealtime),
       'reid' : isChecked(isReid),
       'heatmap' : isChecked(isHeatmap),
       'yolo_weight' : yolo,
       'reid_model' : reid_model,
       'deepsort_model' : ds_model,
       'frame_skip' : frame_skip,
       'video_length' : video_length,
       'heatmap_accumulation' : heatmap_cumul_sec,
       'fps' : fps,
       'videos_num' : videos_num,
       'resolution' : resol,
   };
   setSpinner(true);
   $.ajax({
      type:'POST',
      url:'/',
      data:JSON.stringify(param),
      success:function(json){
          console.log("data pass success", json);
          setSpinner(false);
          $('.output-wrapper').show();
      },
      error:function(){
          alert('data fail');
          setSpinner(false);
      }
   });
});