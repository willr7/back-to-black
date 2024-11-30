from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-40-mini",
    messages=[
        {"role":"system", "content": "You are a helpful African American Vernacular English(AAVE) to Standard Academic English(SAE) translator"},
        {"role":"user", "content": "Translate the following AAVE lyrics where each line is separated by a semicolon to SAE lyrics: Ayy, ayy; Oh, oh-oh; Yeah, yeah, yeah, woah; Ayy, one main girl, fuck a mistress, oh; Heart still hurt, like a wrist slit, oh; I don't fit in, I'm a misfit, oh; First thing on my mind, that's to get rich, oh; One main girl, fuck a mistress, oh; Heart still hurt, like a wrist slit, oh; I don't fit in, I'm a misfit, oh; First thing on my mind, that's to get rich, oh; I'm a misfit, that means I don't fit, nah, I don't fit in; I just bought a coupé, baby girl, get in; Lord forgive me, I know I'ma sin tonight; Prolly get high with my friends tonight; Have a ménage in the Benz tonight; (Yeah); Prolly go and fuck on some twins tonight; Pop a xanax so I'll forget tonight (Yeah, tonight); Went to L.A., almost missed my flight twice; (Yeah, yeah, twice); I was off the xans that night, right (Right); Almost died on 'em, that's the last flight (Flight); I was pretty cool in my last life (Life); Ayy, one main girl, fuck a mistress, oh (Ayy, ayy, yeah); Heart still hurt, like a wrist slit, oh (Yeah); I don't fit in, I'm a misfit, oh (Yeah); First thing on my mind is to get rich, oh; One main girl, fuck a mistress, oh; Heart still hurt, like a wrist slit, oh; I don't fit in, I'm a misfit, oh; First thing on my mind, that's to get rich, oh; Uh, ayy; Xanny turning me into a beast (Uh, uh); Don't make me call them niggas, I'm over the east (Ayy); Then I'll probably push you in the street (Uh, yeah, ayy); Make a motherfucker hit his feet; Tell them niggas, Check my swag Yours too bad, take that back; I quit drugs, then relapsed; I run back, like gym class; Brand new bitch, I need it (Uh, ayy); If it's too good, I'll keep it; Be my Victoria's Secret; Be my Victoria's Secret, yeah; (Ayy); One main girl, fuck a mistress, oh; Heart still hurt, never wrist slit, oh (Yeah); I don't fit in, I'm a misfit, oh; First thing on my mind, that's to get rich, oh; One main girl, fuck a mistress, oh (Ayy); Heart still hurt, never wrist slit, oh; I don't fit in, I'm a misfit, oh; First thing on my mind, that's to get rich, oh"}
    ]
)


print(completion.choices[0].message)