window.addEventListener("load", function(){
    // [1] GET ALL THE HTML ELEMENTS
    var capture_container = document.getElementById("capture_container");
        play_capture = document.getElementById("play_capture");
        save_person_name = document.getElementById("save_person_name");

    play_capture.addEventListener("click", function(){
      console.log('capture call')
      capture_container.insertAdjacentHTML('beforeend', "<img src='/capture_video_pic'>");
    })

    save_person_name.addEventListener("click", function(){
      const person_name = document.getElementById('person_name').value
      $.ajax({
        type:"POST",
        dataType: 'json',
        contentType: 'application/json',
        url: "http://localhost:3000/save-multiple-image",
        data : JSON.stringify({'person_name': person_name}),
        success: function(data){
            buf1=data;
            console.log(data);
        }
      })
    })
  });