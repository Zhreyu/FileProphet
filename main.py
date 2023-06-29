import discord
from discord.ext import commands
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
import os

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

file = None
stored_query = None

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
embeddings = HuggingFaceEmbeddings()
llm6 = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
chain = load_qa_chain(llm6, chain_type="stuff")     

@bot.event
async def on_ready():
    print("Bot is Up and Ready!")

@bot.command()
async def hello(ctx: commands.Context):
    await ctx.send(f"Hey {ctx.author.mention}! This is a command.")

@bot.command()
async def say(ctx: commands.Context, *, thing_to_say: str):
    await ctx.send(f"{ctx.author.name} said: {thing_to_say}")

@bot.command()
async def inputfile(ctx: commands.Context):
    global file
    global stored_query

    if len(ctx.message.attachments) == 0:
        await ctx.send("No file attached.")
        return

    attachment = ctx.message.attachments[0]
    if attachment.filename.endswith('.csv'):
        loader = CSVLoader(file_path=attachment.filename)
    elif attachment.filename.endswith('.pdf'): 
        loader = UnstructuredPDFLoader(file_path=attachment.filename)

    try:
        file = attachment.filename
        await attachment.save(file)
        await ctx.send(f"{file} saved successfully.")

        while True:
            await ctx.send("Please enter a query (or type 'exit' to quit):")
            query_msg = await bot.wait_for('message', check=lambda m: m.author == ctx.author, timeout=60.0)
            stored_query = query_msg.content

            if stored_query.lower() == "exit":
                break
            data = loader.load()
            db = FAISS.from_documents(data, embeddings)

            # Run langchain model on query and retrieve results
            docs = db.similarity_search(stored_query)
            results = chain.run(input_documents=docs, question=stored_query)
            await ctx.send(results)
            await ctx.send("Any further assistance?")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        await ctx.send("Time out")
        



@bot.command()
async def exit(ctx: commands.Context):
    global file
    
    if file is None:
        await ctx.send("No files has been uploaded.")
        return
    
    try:
        os.remove(file)
        file = None
        await ctx.send(" file deleted successfully.")
        
    except:
        await ctx.send("Failed to delete the file.")


bot.run('')
