window.addEventListener("load", function(){
    // [1] GET ALL THE HTML ELEMENTS
    var video = document.getElementById("vid-show"),
        canvas = document.getElementById("vid-canvas"),
        take = document.getElementById("vid-take");
        save = document.getElementById("save-image");
  
    // [2] ASK FOR USER PERMISSION TO ACCESS CAMERA
    // WILL FAIL IF NO CAMERA IS ATTACHED TO COMPUTER
    navigator.mediaDevices.getUserMedia({ video : true })
    .then(function(stream) {
      // [3] SHOW VIDEO STREAM ON VIDEO TAG
      video.srcObject = stream;
      video.play();
  
      // [4] WHEN WE CLICK ON "TAKE PHOTO" BUTTON
      take.addEventListener("click", function(){
        // Create snapshot from video
        var draw = document.createElement("canvas");
        draw.id='canvas-image';
        draw.width = video.videoWidth;
        draw.height = video.videoHeight;
        var context2D = draw.getContext("2d");
        context2D.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        // Put into canvas container
        canvas.innerHTML = "";


        canvas.appendChild(draw);
      });
    })
    .catch(function(err) {
      document.getElementById("vid-controls").innerHTML = "Please enable access and attach a camera";
    });


    save.addEventListener("click", function(){
        console.log('work')
        const canvas = document.getElementById("canvas-image");
        const img    = canvas.toDataURL("image/png");
        const person_name = document.getElementById('person_name').value
        // var src = $('#canvasId').toDataURL("image/png");
        console.log('draw', img)
        $.ajax({
            type:"POST",
            dataType: 'json',
            contentType: 'application/json',
            url: "http://localhost:3000/save-image",
            data : JSON.stringify({'img_path': img, 'person_name': person_name}),
            success: function(data){
                buf1=data;
                console.log(data);
            }
        })
    })
  });