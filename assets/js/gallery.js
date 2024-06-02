// script.js 
function changeReadMore() { 
    const mycontent = 
        document.getElementById('mybox1id'); 
    const mybutton = 
        document.getElementById('mybuttonid'); 
  
    if (mycontent.style.display === 'none'
        || mycontent.style.display === '') { 
        mycontent.style.display = 'block'; 
        mybutton.textContent = 'Späť do galérie'; 
    } else { 
        mycontent.style.display = 'none'; 
        mybutton.innerHTML = '<img src="images/460.png" alt="buttonpng" border="0" width="100" />'; 
    } 
}

// script.js 
function changeReadMore_2() { 
    const mycontent = 
        document.getElementById('mybox2id'); 
    const mybutton = 
        document.getElementById('mybuttonid_2'); 
  
    if (mycontent.style.display === 'none'
        || mycontent.style.display === '') { 
        mycontent.style.display = 'block'; 
        mybutton.textContent = 'Späť do galérie'; 
    } else { 
        mycontent.style.display = 'none'; 
        mybutton.innerHTML = '<img src="images/elephant.jpg" alt="buttonpng" border="0" width="100" />'; 
    } 
}
