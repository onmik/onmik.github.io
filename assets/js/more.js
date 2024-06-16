function changeReadMore() { 
    const mycontent = 
        document.getElementById('mybox1id'); 
    const mybutton = 
        document.getElementById('mybuttonid'); 
    const span1 = document.getElementById("span1") 
  
    if (mycontent.style.display === 'none'
        || mycontent.style.display === '') { 
        mycontent.style.display = 'inline'; 
        span1.style.display = "none"; 
        mybutton.textContent = 'Zbaliť článok'; 
    } else { 
        mycontent.style.display = 'none'; 
        mybutton.textContent = 'Čítať ďalej'; 
        span1.style.display = "inline"; 
    } 
} 

function changeReadMore_2() { 
    const mycontent = 
        document.getElementById('mybox2id'); 
    const mybutton = 
        document.getElementById('mybutton2id'); 
    const span1 = document.getElementById("span2") 
  
    if (mycontent.style.display === 'none'
        || mycontent.style.display === '') { 
        mycontent.style.display = 'inline'; 
        span1.style.display = "none"; 
        mybutton.textContent = 'Zbaliť článok'; 
    } else { 
        mycontent.style.display = 'none'; 
        mybutton.textContent = 'Čítať ďalej'; 
        span1.style.display = "inline"; 
    } 
} 
